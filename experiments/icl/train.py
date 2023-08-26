"""
training the transformer on synthetic in-context regression task
"""

import logging
import os
import random
import warnings
from typing import Optional

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
    return (ys_true - ys_pred).square().mean(axis=axis) ** 0.5

def accuracy_fn(ys_true, ys_pred, axis=None, atol=1e-3):
    return ((ys_true - ys_pred).abs() < atol).mean(axis=axis, dtype=torch.float64) 
    
@torch.no_grad()
def evals(model, data, num_examples, batch_size, ):
    xs, ys = data.get_batch(
        num_examples=num_examples,
        batch_size=batch_size,
    )
    
    preds = {
        'model': model(xs, ys),
        'dmmse': dmmse_predictor(
            xs,
            ys,
            prior=data.task_distribution,
            noise_variance=data.noise_variance,
        ),
        'ridge': ridge_predictor(
            xs,
            ys,
            noise_variance=data.noise_variance,
        ),
    }
    metrics = {f"{alg}/{type_}/{metric}": 0.0 for alg in preds.keys() for metric in ["per_token", "avg", "final"] for type_ in ["mse", "acc"]}

    for m, ys_ in preds.items():
        per_token_losses = loss_fn(ys, ys_, axis=(0,2))
        metrics[f"{m}/mse/per_token"] = per_token_losses.tolist()
        metrics[f"{m}/mse/avg"] = per_token_losses.mean().item()
        metrics[f"{m}/mse/final"] = per_token_losses[-1].item()

        per_token_accuracies = accuracy_fn(ys, ys_, axis=(0,2))
        metrics[f"{m}/acc/per_token"] = per_token_accuracies.tolist()
        metrics[f"{m}/acc/avg"] = per_token_accuracies.mean().item()
        metrics[f"{m}/acc/final"] = per_token_accuracies[-1].item()

    return metrics


def train(config: ICLConfig, seed=0, is_debug=False):
    logging.basicConfig(level=logging.INFO if not is_debug else logging.DEBUG)

    set_seed(seed)

    data = RegressionSequenceDistribution(
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
    checkpointer = CheckpointManager("icl", "devinterp") # , "experiments/icl")
    logger = Logger(config.project, config.entity, config.logging_steps)

    print(checkpointer, config.checkpoint_steps)

    def state_dict():
        return {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler else None,
        }

    for step in tqdm.trange(config.num_steps, desc=f"Epoch 0 Batch 0/{config.num_steps} Loss: ?.??????"):
        set_seed(seed + step)  # For reproducibility if we resume training

        # data generation and forward pass
        xs, ys = data.get_batch(
            num_examples=config.max_examples,
            batch_size=config.batch_size,
        )
        ys_pred = model(xs, ys)
        loss = loss_fn(ys_true=ys, ys_pred=ys_pred)
        # backward pass and gradient step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if config.is_wandb_enabled:
            # TODO: Figure out how to make this work with Logger
            wandb.log({"batch/loss": loss.item()}, step=step)

        # Log to wandb & save checkpoints according to log_steps
        if step in config.checkpoint_steps:
            print("saving checkpoint")
            logger.info(f"Saving checkpoint at step {step}")
            checkpointer.save_file((0, step), state_dict())

        if step in config.logging_steps:
            logger.log(evals(model, data, config.max_examples, config.eval_batch_size), step=step)
            model.train()

    if config.is_wandb_enabled:
        wandb.finish()


def get_config(project=None, entity=None):
    use_wandb = project is not None and entity is not None

    if use_wandb:
        wandb.init(project=project, entity=entity)

    config_dict = {
        "num_steps": 524_288, # for the paper (500k)
        "num_training_samples": 524_288 * 256,
        "batch_size": 256,
        "logging_steps": None, # (1000, 1000), 
        "checkpoint_steps": (100, 100),
        "optimizer_config": {
            "optimizer_type": "Adam",
            "betas": (0.9, 0.999),
            "weight_decay": 0.0,
            "lr": 1e-3,
        },
        "project": project,
        "entity": entity,
    }

    if use_wandb:
        config_dict.update(wandb.config)

    return ICLConfig(**config_dict)

if __name__ == "__main__":
    # config = get_config(project="devinterp", entity="devinterp")
    config = get_config()
    train(config, seed=0, is_debug=False)


    
    

