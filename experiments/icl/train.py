"""
training the transformer on synthetic in-context regression task
"""

import functools
import logging
import os
import random
import warnings
from typing import Dict, List, Optional, TypedDict

import numpy as np
import torch
import tqdm
#
from baselines import dmmse_predictor, ridge_predictor
from dotenv import load_dotenv
from model import InContextRegressionTransformer
from tasks import (DiscreteTaskDistribution, GaussianTaskDistribution,
                   RegressionSequenceDistribution)

import wandb
from devinterp.config import Config
from devinterp.logging import Logger
from devinterp.storage import CheckpointManager
from devinterp.utils import flatten_dict

load_dotenv()

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = "1"

class ICLConfig(Config):
    # Dataset & loader
    task_size: int = 8
    max_examples: int = 16
    num_tasks: int = 64
    noise_variance: float = 0.25
    eval_batch_size: int = 2048
    
    # Model
    embed_size: int = 128
    mlp_size: int = 128
    num_heads: int = 2 
    num_layers: int = 8
    

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    try:
        torch.cuda.manual_seed_all(seed) 
    except AttributeError:
        warnings.warn("CUDA not available")


def loss_fn(ys_true, ys_pred, axis=None):
    return (ys_true - ys_pred).square().mean(axis=axis) 


class LossSummary(TypedDict):
    per_token: List[float]
    avg: float
    last: float

def losses_fn(ys, yhats) -> LossSummary:
    losses = loss_fn(ys, yhats, axis=(0,2))

    return {
        "per_token": losses.tolist(),
        "avg": losses.mean().item(),
        "last": losses[-1].item(),
    }

class StateDict(TypedDict):
    model: Dict
    optimizer: Dict
    scheduler: Optional[Dict]


def state_dict(model, optimizer, scheduler) -> StateDict:
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
    }


def train(config: ICLConfig, seed=0, is_debug=False):
    logging.basicConfig(level=logging.INFO if not is_debug else logging.DEBUG)
    set_seed(seed)

    # initialise model
    model = InContextRegressionTransformer(
        task_size=config.task_size,
        max_examples=config.max_examples,
        embed_size=config.embed_size,
        mlp_size=config.mlp_size,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        device=config.device,
    )
    model.to(config.device)
    model.train()

    # initialise training stuff
    optimizer = config.optimizer_config.factory(model.parameters())
    scheduler = config.scheduler_config.factory(optimizer) if config.scheduler_config else None

    # TODO: hash the model details to create a unique identifier for the model and use this as project name below
    checkpointer = CheckpointManager(f"icl-ntasks-{config.num_tasks}", "devinterp") # , "experiments/icl")
    logger = Logger(config.project, config.entity, config.logging_steps)

    # initialise data sources
    pretrain_dist = RegressionSequenceDistribution(
        task_distribution=DiscreteTaskDistribution(
            num_tasks=config.num_tasks,
            task_size=config.task_size,
            device=config.device,
        ),
        noise_variance=config.noise_variance,
    )

    true_dist = RegressionSequenceDistribution(
        task_distribution=GaussianTaskDistribution(
            task_size=config.task_size,
            device=config.device,
        ),
        noise_variance=config.noise_variance,
    )

    # initialise evaluation stuff
    # fixed evaluation data batches
    pretrain_data = pretrain_dist.get_batch(
        num_examples=config.max_examples,
        batch_size=config.eval_batch_size,
    )
    pretrain_dmmse_preds = dmmse_predictor(*pretrain_data, pretrain_dist.task_distribution, pretrain_dist.noise_variance)
    pretrain_ridge_preds = ridge_predictor(*pretrain_data, config.noise_variance)

    # For the following, we use the true distribution to generate the data, but the pretrain distribution to "train" the predictor
    true_data = true_dist.get_batch(
        num_examples=config.max_examples,
        batch_size=config.eval_batch_size,
    )
    true_dmmse_preds = dmmse_predictor(*true_data, pretrain_dist.task_distribution, pretrain_dist.noise_variance)
    true_ridge_preds = ridge_predictor(*true_data, config.noise_variance)

    # to evaluate a model on the batches against these baselines
    @torch.no_grad()
    def evals():
        pretrain_model_preds = model(*pretrain_data)
        pretrain_model_losses = losses_fn(pretrain_data[1], pretrain_model_preds)
        pretrain_delta_model_vs_dmmse = loss_fn(pretrain_model_preds, pretrain_dmmse_preds)
        pretrain_delta_model_vs_ridge = loss_fn(pretrain_model_preds, pretrain_ridge_preds)
        
        true_model_preds = model(*true_data)
        true_model_losses = losses_fn(true_data[1], true_model_preds)
        true_delta_model_vs_dmmse = loss_fn(true_model_preds, true_dmmse_preds)
        true_delta_model_vs_ridge = loss_fn(true_model_preds, true_ridge_preds)

        return {
            "pretrain/per_token": pretrain_model_losses["per_token"],
            "pretrain/mse": pretrain_model_losses["avg"],
            "pretrain/last_token": pretrain_model_losses["last"],
            "pretrain/delta_dmmse": pretrain_delta_model_vs_dmmse,
            "pretrain/delta_ridge": pretrain_delta_model_vs_ridge,
            "true/per_token": true_model_losses["per_token"],
            "true/mse": true_model_losses["avg"],
            "true/last_token": true_model_losses["last"],
            "true/delta_dmmse": true_delta_model_vs_dmmse,
            "true/delta_ridge": true_delta_model_vs_ridge,
        }


    for step in tqdm.trange(config.num_steps, desc=f"Epoch 0 Batch 0/{config.num_steps} Loss: ?.??????"):
        set_seed(seed + step)  # For reproducibility if we resume training

        # data generation and forward pass
        xs, ys = pretrain_dist.get_batch(
            num_examples=config.max_examples,
            batch_size=config.batch_size,
        )
        ys_pred = model(xs, ys)
        loss = loss_fn(ys_true=ys, ys_pred=ys_pred)
        # backward pass and gradient step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if scheduler:
            scheduler.step()

        if config.is_wandb_enabled:
            # TODO: Figure out how to make this work with Logger
            wandb.log({"batch/loss": loss.item()}, step=step)

        # Log to wandb & save checkpoints according to log_steps
        if step in config.checkpoint_steps:
            print("saving checkpoint")
            logger.info(f"Saving checkpoint at step {step}")
            checkpointer.save_file((0, step), state_dict(model, optimizer, scheduler))

        if step in config.logging_steps:
            logger.log(evals(), step=step)
            model.train()

    if config.is_wandb_enabled:
        wandb.finish()


def get_config(project=None, entity=None):
    use_wandb = project is not None and entity is not None

    if use_wandb:
        wandb.init(project=project, entity=entity)

    num_steps = 524_288 # for the paper (500k)
    batch_size = 256
    max_learning_rate = 1e-3
    config_dict = {
        "num_steps": num_steps, 
        "num_training_samples": num_steps * batch_size,
        "batch_size": batch_size,
        "logging_steps": (500, 500), 
        "checkpoint_steps": (100, 100),
        "optimizer_config": {
            "optimizer_type": "Adam",
            "betas": (0.9, 0.999),
            "weight_decay": 0.0,
            "lr": max_learning_rate,    # unused (overwritten by scheduler)
        },
        "scheduler_config": {
            "scheduler_type": "OneCycleLR",
            "max_lr": max_learning_rate,
            "total_steps": num_steps,
            "anneal_strategy": 'linear',
            "div_factor": (num_steps/2 - 1),        # start 1 step past 0
            "final_div_factor": (num_steps/2 - 1),  # end 1 step before 0
            "pct_start": 0.5,           # 50% warmup
            "cycle_momentum": False,    # Adam doesn't support momentum
        },
        "project": project,
        "entity": entity,
    }

    if use_wandb:
        config_dict.update(wandb.config)

    return ICLConfig(**config_dict)

if __name__ == "__main__":
    config = get_config(project="icl", entity="devinterp")
    # config = get_config()
    train(config, seed=0, is_debug=False)

