import functools
import random
import warnings
from typing import Callable, Container, Dict, List, Optional, Tuple, TypedDict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import wandb
from torch.optim.lr_scheduler import LambdaLR
from tqdm.notebook import tqdm

from devinterp.storage import CheckpointManager
from devinterp.config import Config
from devinterp.logging import Logger

wandb.finish()

class LearnerStateDict(TypedDict):
    model: Dict
    optimizer: Dict
    scheduler: Optional[Dict]

class Learner:
    def __init__(self, model: torch.nn.Module, train_set: torch.utils.data.Dataset, test_set: torch.utils.data.DataLoader, config: Config, metrics: Optional[List[Callable[['Learner'], Dict]]]=None):
        self.config = config
        self.model = model
        self.train_set = train_set
        self.test_set = test_set
        self.optimizer = config.optimizer_config.factory(model.parameters())

        def lr_lambda(step: int):
            if step < 400:
                return 0.1
            elif step < 32_000:
                return 1.
            elif step < 48_000:
                return 0.1
            else:
                return 0.01

        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lr_lambda) # config.scheduler_config.factory(self.optimizer)
        self.generator = torch.Generator(device="cpu")
        self.sampler = torch.utils.data.RandomSampler(train_set, generator=self.generator)
        self.train_loader = torch.utils.data.DataLoader(train_set, batch_size=config.batch_size, sampler=self.sampler)
        self.test_loader = torch.utils.data.DataLoader(test_set, batch_size=config.batch_size, shuffle=False)
        self.metrics = metrics or []
        self.logger = Logger(project=config.project, entity=config.entity, logging_steps=config.logging_steps, metrics=[], out_file=None, use_df=False)
        self.checkpoints = CheckpointManager(f"{model.__class__.__name__}18/{self.train_loader.dataset.__class__.__name__}", 'devinterp')  # TODO: read 18 automatically
        
    def measure(self):
        return functools.reduce(lambda x, y: x | y, [metric(self) for metric in self.metrics], {})

    def resume(self, batch_idx: Optional[int] = None):
        if batch_idx is None:
            epoch, batch_idx = self.checkpoints[-1]
        else:
            epoch, batch = min(self.checkpoints, key=lambda x: abs(x[1] - batch_idx))

            if batch != batch_idx:
                warnings.warn(f"Could not find checkpoint with batch_idx {batch_idx}. Resuming from closest batch ({batch}) instead.")

        self.load(epoch, batch_idx)

    def train(self, resume=False, run_id: Optional[str] = None):
        if resume:
            self.resume(resume, run_id)

            if self.scheduler:
                self.scheduler.last_epoch = self.config.num_steps_per_epoch * epoch + batch_idx

        self.model.to(self.config.device)
        self.model.train()

        if self.config.is_wandb_enabled:
            if resume and not run_id:
                warnings.warn("Resuming from checkpoint but no run_id provided. Will not log to existing wandb run.")
            
            if not run_id:
                wandb.init(project=self.config.project, entity=self.config.entity)
            else:
                wandb.init(project=self.config.project, entity=self.config.entity, run_id=run_id)

        pbar = tqdm(total=self.config.num_steps, desc=f"Epoch 0 Batch 0/{self.config.num_steps} Loss: ?.??????")
        
        for epoch in range(0, self.config.num_epochs):
            self.set_seed(epoch)

            for _batch_idx, (data, target) in enumerate(self.train_loader):
                batch_idx = self.config.num_steps_per_epoch * epoch + _batch_idx
                data, target = data.to(self.config.device), target.to(self.config.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = F.cross_entropy(output, target)
                loss.backward()
                self.optimizer.step()

                if self.scheduler:
                    self.scheduler.step()

                # Update progress bar description
                pbar.set_description(f"Epoch {epoch} Batch {batch_idx}/{self.config.num_steps} Loss: {loss.item():.6f}")
                pbar.update(1)

                if self.config.is_wandb_enabled:
                    # TODO: Figure out how to make this work with Logger
                    wandb.log({"Batch/Loss": loss.item()}, step=batch_idx)

                # Log to wandb & save checkpoints according to log_steps
                if batch_idx in self.config.checkpoint_steps:
                    self.save_checkpoint(epoch, batch_idx)

                if batch_idx in self.config.logging_steps:
                    self.logger.log(self.measure(), step=batch_idx)
                    self.model.train()

            pbar.close()

        if self.config.is_wandb_enabled:
            wandb.finish()

    def state_dict(self) -> LearnerStateDict:
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
        }

    def load_state_dict(self, checkpoint: LearnerStateDict):
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])

        if self.scheduler is not None and checkpoint["scheduler"] is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler"])
    
    def save_checkpoint(self, epoch: int, batch_idx: int):
        checkpoint = self.state_dict()
        self.checkpoints.save_checkpoint(checkpoint, epoch, batch_idx)

    def load_checkpoint(self, epoch: int, batch_idx: int):
        checkpoint = self.checkpoints.load_checkpoint(epoch, batch_idx)
        self.load_state_dict(checkpoint)

    def set_seed(self, seed: int):
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        self.generator.manual_seed(seed)

        if "cuda" in str(self.config.device):
            torch.cuda.manual_seed_all(seed) 
    