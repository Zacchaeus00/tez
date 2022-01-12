"""
The tez model class
"""

import os
import warnings
from dataclasses import dataclass

import multiprocessing
import torch
import torch.nn as nn
from tez import enums
from tez.callbacks import CallbackRunner
from tez.utils import AverageMeter
from tqdm.auto import tqdm

try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp

    XLA_AVAILABLE = True
except ImportError:
    XLA_AVAILABLE = False


@dataclass
class Model:
    model = None
    train_loader = None
    valid_loader = None
    optimizer = None
    scheduler = None
    step_scheduler_after = None
    step_scheduler_metric = None
    current_epoch = 0
    current_train_step = 0
    current_valid_step = 0
    _model_state = None
    _train_state = None
    device = None
    _callback_runner = None
    fp16 = False
    scaler = None
    accumulation_steps = 0
    batch_index = 0
    metrics = {}
    metrics["train"] = {}
    metrics["valid"] = {}
    metrics["test"] = {}
    clip_grad_norm = None
    using_tpu = False
    local_rank = -1
    train_sampler = None
    valid_sampler = None

    @property
    def model_state(self):
        return self._model_state

    @model_state.setter
    def model_state(self, value):
        self._model_state = value
        # run something here in future if needed

    @property
    def train_state(self):
        return self._train_state

    @train_state.setter
    def train_state(self, value):
        self._train_state = value
        if self._callback_runner is not None:
            self._callback_runner(value)

    def name_to_metric(self, metric_name):
        if metric_name == "current_epoch":
            return self.current_epoch
        v_1 = metric_name.split("_")[0]
        v_2 = "_".join(metric_name.split("_")[1:])
        return self.metrics[v_1][v_2]

    def monitor_metrics(self, *args, **kwargs):
        return

    def loss(self, *args, **kwargs):
        return

    def fetch_optimizer(self, *args, **kwargs):
        return

    def fetch_scheduler(self, *args, **kwargs):
        return

    def model_fn(self, data):
        for key, value in data.items():
            data[key] = value.to(self.device)
        if self.fp16:
            with torch.cuda.amp.autocast():
                op = self.model(**data)
        else:
            op = self.model(**data)
        output, loss = op["logits"], op["loss"]
        if self.local_rank == -1:
            if self.device_count > 1:
                loss = loss.mean()
        return output, loss, {}

    def train_one_step(self, data):
        if self.accumulation_steps == 1 and self.batch_index == 0:
            self.model.zero_grad()
        _, loss, metrics = self.model_fn(data)
        loss = loss / self.accumulation_steps
        if self.fp16:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        if self.clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
        if (self.batch_index + 1) % self.accumulation_steps == 0:
            if self.fp16:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                if self.using_tpu:
                    xm.optimizer_step(self.optimizer, barrier=True)
                else:
                    self.optimizer.step()
            if self.scheduler:
                if self.step_scheduler_after == "batch":
                    if self.step_scheduler_metric is None:
                        self.scheduler.step()
                    else:
                        step_metric = self.name_to_metric(self.step_scheduler_metric)
                        self.scheduler.step(step_metric)
            if self.batch_index > 0:
                self.model.zero_grad()
        return loss, metrics

    def validate_one_step(self, data):
        _, loss, metrics = self.model_fn(data)
        if self.local_rank == -1:
            if self.device_count > 1:
                loss = loss.mean()
        return loss, metrics

    def predict_one_step(self, data):
        output, _, _ = self.model_fn(data)
        return output

    def update_metrics(self, losses, monitor):
        self.metrics[self._model_state.value].update(monitor)
        self.metrics[self._model_state.value]["loss"] = losses.avg

    def train_one_epoch(self, data_loader):
        if self.local_rank != -1:
            self.train_sampler.set_epoch(self.current_epoch)

        self.model.train()
        if self.local_rank != -1:
            torch.distributed.barrier()

        self.model_state = enums.ModelState.TRAIN
        losses = AverageMeter()
        if self.accumulation_steps > 1:
            self.optimizer.zero_grad()
        if self.using_tpu:
            tk0 = data_loader
        else:
            tk0 = tqdm(data_loader, total=len(data_loader))
        for b_idx, data in enumerate(tk0):
            self.batch_index = b_idx
            self.train_state = enums.TrainingState.TRAIN_STEP_START
            loss, metrics = self.train_one_step(data)
            self.train_state = enums.TrainingState.TRAIN_STEP_END
            losses.update(loss.item() * self.accumulation_steps, data_loader.batch_size)
            if b_idx == 0:
                metrics_meter = {k: AverageMeter() for k in metrics}
            monitor = {}
            for m_m in metrics_meter:
                metrics_meter[m_m].update(metrics[m_m], data_loader.batch_size)
                monitor[m_m] = metrics_meter[m_m].avg
            self.current_train_step += 1
            if not self.using_tpu:
                tk0.set_postfix(loss=losses.avg, stage="train", **monitor)
            if self.using_tpu:
                print(f"train step: {self.current_train_step} loss: {losses.avg}")
        if not self.using_tpu:
            tk0.close()
        self.update_metrics(losses=losses, monitor=monitor)

        return losses.avg

    def validate_one_epoch(self, data_loader):
        self.model.eval()
        self.model_state = enums.ModelState.VALID
        losses = AverageMeter()
        if self.using_tpu:
            tk0 = data_loader
        else:
            tk0 = tqdm(data_loader, total=len(data_loader))
        for b_idx, data in enumerate(tk0):
            self.train_state = enums.TrainingState.VALID_STEP_START
            with torch.no_grad():
                loss, metrics = self.validate_one_step(data)
                if self.local_rank != -1:
                    torch.distributed.barrier()
                    output_tensors = [loss.clone() for _ in range(torch.distributed.get_world_size())]
                    torch.distributed.all_gather(output_tensors, loss)
                    loss = torch.mean(torch.stack(output_tensors))

            self.train_state = enums.TrainingState.VALID_STEP_END
            losses.update(loss.item(), data_loader.batch_size)
            if b_idx == 0:
                metrics_meter = {k: AverageMeter() for k in metrics}
            monitor = {}
            for m_m in metrics_meter:
                metrics_meter[m_m].update(metrics[m_m], data_loader.batch_size)
                monitor[m_m] = metrics_meter[m_m].avg
            if not self.using_tpu:
                tk0.set_postfix(loss=losses.avg, stage="valid", **monitor)
            self.current_valid_step += 1
        if not self.using_tpu:
            tk0.close()
        self.update_metrics(losses=losses, monitor=monitor)
        return losses.avg

    def process_output(self, output):
        output = output.cpu().detach().numpy()
        return output

    def predict(self, dataset, sampler=None, batch_size=16, n_jobs=1, collate_fn=None):
        if next(self.parameters()).device != self.device:
            self.to(self.device)

        if n_jobs == -1:
            n_jobs = multiprocessing.cpu_count()

        if batch_size == 1:
            n_jobs = 0
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, num_workers=n_jobs, sampler=sampler, collate_fn=collate_fn, pin_memory=True
        )

        if self.model.training:
            self.model.eval()

        if self.using_tpu:
            tk0 = data_loader
        else:
            tk0 = tqdm(data_loader, total=len(data_loader))

        for _, data in enumerate(tk0):
            with torch.no_grad():
                out = self.predict_one_step(data)
                out = self.process_output(out)
                yield out

            if not self.using_tpu:
                tk0.set_postfix(stage="test")

        if not self.using_tpu:
            tk0.close()

    def save(self, model_path, weights_only=False):
        model_state_dict = self.model.state_dict()
        if weights_only:
            if self.using_tpu:
                xm.save(model_state_dict, model_path)
            else:
                torch.save(model_state_dict, model_path)
            return
        if self.optimizer is not None:
            opt_state_dict = self.optimizer.state_dict()
        else:
            opt_state_dict = None
        if self.scheduler is not None:
            sch_state_dict = self.scheduler.state_dict()
        else:
            sch_state_dict = None
        model_dict = {}
        model_dict["state_dict"] = model_state_dict
        model_dict["optimizer"] = opt_state_dict
        model_dict["scheduler"] = sch_state_dict
        model_dict["epoch"] = self.current_epoch
        model_dict["fp16"] = self.fp16
        if self.using_tpu:
            xm.save(model_dict, model_path)
        else:
            torch.save(model_dict, model_path)

    def load(self, model_path, weights_only=False, device="cuda"):
        if device == "tpu":
            if XLA_AVAILABLE is False:
                raise RuntimeError("XLA is not available")
            else:
                self.using_tpu = True
                device = xm.xla_device()
        self.device = device
        if next(self.model.parameters()).device != self.device:
            self.model.to(self.device)
        model_dict = torch.load(model_path, map_location=torch.device(device))
        if weights_only:
            self.model.load_state_dict(model_dict)
        else:
            self.model.load_state_dict(model_dict["state_dict"])

    def _init_tez(
        self,
        args,
        train_dataset,
        valid_dataset,
        train_sampler,
        valid_sampler,
        callbacks,
        train_collate_fn,
        valid_collate_fn,
    ):

        self.train_sampler = train_sampler
        self.valid_sampler = valid_sampler

        if callbacks is None:
            callbacks = list()

        if args.n_jobs == -1:
            n_jobs = multiprocessing.cpu_count()

        self.accumulation_steps = args.accumulation_steps
        self.clip_grad_norm = args.clip_grad_norm
        self.device_count = torch.cuda.device_count()
        if self.local_rank == -1:
            if self.device_count > 1:
                self.model = nn.DataParallel(self.model)
        else:
            torch.distributed.init_process_group(backend="nccl")
            self.model = nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=True,
            )
            if self.train_sampler is None:
                self.train_sampler = torch.utils.data.distributed.DistributedSampler(
                    train_dataset,
                    shuffle=args.train_shuffle,
                )

        if self.train_loader is None:
            self.train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=args.train_batch_size,
                num_workers=n_jobs,
                sampler=train_sampler,
                shuffle=args.train_shuffle if train_sampler is None else False,
                collate_fn=train_collate_fn,
            )
        if self.valid_loader is None:
            if valid_dataset is not None:
                self.valid_loader = torch.utils.data.DataLoader(
                    valid_dataset,
                    batch_size=args.valid_batch_size,
                    num_workers=n_jobs,
                    sampler=valid_sampler,
                    shuffle=args.train_shuffle if train_sampler is None else False,
                    collate_fn=valid_collate_fn,
                )

        if self.optimizer is None:
            self.optimizer = self.fetch_optimizer()

        if self.scheduler is None:
            self.scheduler = self.fetch_scheduler()

        self.fp16 = args.fp16
        if self.fp16:
            self.scaler = torch.cuda.amp.GradScaler()

        self._callback_runner = CallbackRunner(callbacks, self)
        self.train_state = enums.TrainingState.TRAIN_START

    def fit(
        self,
        train_dataset,
        args,
        valid_dataset=None,
        train_sampler=None,
        valid_sampler=None,
        callbacks=None,
        train_collate_fn=None,
        valid_collate_fn=None,
    ):
        self.local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
        if args.device == "tpu":
            if XLA_AVAILABLE is False:
                raise RuntimeError("XLA is not available")
            else:
                self.using_tpu = True
                args.fp16 = False
                self.device = xm.xla_device()

        elif args.device.startswith("cuda"):
            if self.local_rank == -1:
                self.device = torch.device(args.device)
            else:
                self.device = torch.device("cuda", self.local_rank)
        else:
            self.device = torch.device("cpu")

        self.model.to(self.device)
        self._init_tez(
            args,
            train_dataset,
            valid_dataset,
            train_sampler,
            valid_sampler,
            callbacks,
            train_collate_fn,
            valid_collate_fn,
        )
        for _ in range(args.epochs):
            self.train_state = enums.TrainingState.EPOCH_START
            self.train_state = enums.TrainingState.TRAIN_EPOCH_START
            _ = self.train_one_epoch(self.train_loader)
            self.train_state = enums.TrainingState.TRAIN_EPOCH_END
            if self.valid_loader:
                self.train_state = enums.TrainingState.VALID_EPOCH_START
                _ = self.validate_one_epoch(self.valid_loader)
                self.train_state = enums.TrainingState.VALID_EPOCH_END
            if self.scheduler:
                if self.step_scheduler_after == "epoch":
                    if self.step_scheduler_metric is None:
                        self.scheduler.step()
                    else:
                        step_metric = self.name_to_metric(self.step_scheduler_metric)
                        self.scheduler.step(step_metric)
            self.train_state = enums.TrainingState.EPOCH_END
            if self._model_state.value == "end":
                break
            self.current_epoch += 1
        self.train_state = enums.TrainingState.TRAIN_END
