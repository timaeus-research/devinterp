"""
training the transformer on synthetic in-context regression task
"""

import functools
import logging
import os
import random
import warnings
from typing import Optional

import numpy as np
import torch
import tqdm
import wandb
#
from baselines import dmmse_predictor, ridge_predictor
from dotenv import load_dotenv
from model import InContextRegressionTransformer
from tasks import (DiscreteTaskDistribution, GaussianTaskDistribution,
                   RegressionSequenceDistribution)

from devinterp.config import Config
from devinterp.logging import Logger
from devinterp.storage import CheckpointManager

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
    embed_size: int = 64 # 128 in the paper
    mlp_size: int = 64 # 128 in the paper
    num_heads: int = 2 # 2 in the paper
    num_layers: int = 2 # 8 in the paper
    

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

def losses_fn(ys, yhats):
    losses = loss_fn(ys, yhats, axis=(0,2))

    return {
        "per_token": losses.tolist(),
        "avg": losses.mean().item(),
        "last": losses[-1].item(),
    }

def compute_dict_deltas(d1, d2):
    return {k: torch.abs(v - d2[k]) for k, v in d1.items()}
    
def prepend_keys(d, prefix, delimiter="/"):
    return {f"{prefix}{delimiter}{k}": v for k, v in d.items()}

@torch.no_grad()
def predictor_loss_evals(xs, ys, data, predictor):
    yhats = predictor(
        xs,
        ys,
        prior=data.task_distribution,
        noise_variance=data.noise_variance,
    )
    return losses_fn(ys, yhats)

dmmse_loss_evals = functools.partial(predictor_loss_evals, predictor=dmmse_predictor)
ridge_loss_evals = functools.partial(predictor_loss_evals, predictor=ridge_predictor)
# model_loss_evals = functools.partial(predictor_loss_evals, predictor=model)

def train(config: ICLConfig, seed=0, is_debug=False):
    logging.basicConfig(level=logging.INFO if not is_debug else logging.DEBUG)

    set_seed(seed)

    pretrain_data = RegressionSequenceDistribution(
        task_distribution=DiscreteTaskDistribution(
            num_tasks=config.num_tasks,
            task_size=config.task_size,
            device=config.device,
        ),
        noise_variance=config.noise_variance,
    )

    true_data = RegressionSequenceDistribution(
        task_distribution=DiscreteTaskDistribution(
            num_tasks=config.num_tasks,
            task_size=config.task_size,
            device=config.device,
        ),
        noise_variance=config.noise_variance,
    )

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

    optimizer = config.optimizer_config.factory(model.parameters())
    scheduler = config.scheduler_config.factory(optimizer) if config.scheduler_config else None

    # TODO: hash the model details to create a unique identifier for the model and use this as project name below
    checkpointer = CheckpointManager(f"icl-ntasks-{config.num_tasks}", "devinterp") # , "experiments/icl")
    logger = Logger(config.project, config.entity, config.logging_steps)

    pretrain_xs, pretrain_ys = pretrain_data.get_batch(
        num_examples=config.max_examples,
        batch_size=8192,
    )

    true_xs, true_ys = true_data.get_batch(
        num_examples=config.max_examples,
        batch_size=8192,
    )

    ref_metrics = {
        "pretrain": {
            "dmmse": dmmse_loss_evals(pretrain_xs, pretrain_ys, pretrain_data),
            "ridge": ridge_loss_evals(pretrain_xs, pretrain_xs, pretrain_data),
        },
        "true": {
            "dmmse": dmmse_loss_evals(true_xs, true_ys, pretrain_data),  # Note: using pretrain_data here is intentional
            "ridge": ridge_loss_evals(true_xs, true_ys, pretrain_data),
        },
    }

    @torch.no_grad()
    def evals():
        def _evals(data, key: str, _ref_metrics):
            xs, ys = data.get_batch(
                num_examples=config.max_examples,
                batch_size=config.eval_batch_size,
            )
            yhats = model(xs, ys)
            losses = losses_fn(ys, yhats)
            delta_dmmse = prepend_keys(compute_dict_deltas(losses, _ref_metrics), "delta_dmmse")
            delta_ridge = prepend_keys(compute_dict_deltas(losses, _ref_metrics), "delta_ridge") 
            losses = prepend_keys(losses, "mse")       
            metrics = losses | delta_dmmse | delta_ridge

            return prepend_keys(metrics, key)

        return _evals(pretrain_data, "pretrain", ref_metrics["pretrain"]) | _evals(true_data, "true", ref_metrics["true"])


    def state_dict():
        return {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler else None,
        }

    for step in tqdm.trange(config.num_steps, desc=f"Epoch 0 Batch 0/{config.num_steps} Loss: ?.??????"):
        set_seed(seed + step * 0.1)  # For reproducibility if we resume training

        # data generation and forward pass
        xs, ys = pretrain_data.get_batch(
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
            checkpointer.save_file((0, step), state_dict())

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
        "checkpoint_steps": None, # was: (100, 100),
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

