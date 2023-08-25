"""
training the transformer on synthetic in-context regression task
"""

# for using torch.linalg.solve on MPS
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = "1"
# (must be done before importing torch)

import tqdm
import torch
torch.manual_seed(42)
from model import InContextRegressionTransformer
from tasks import RegressionSequenceDistribution
from tasks import DiscreteTaskDistribution, GaussianTaskDistribution
from baselines import dmmse_predictor, ridge_predictor


# # COMPUTE SETTINGS

# DEVICE = 'cpu'
DEVICE = 'mps'

# # DATA SETTINGS

INPUT_DIMENSION     =  8
MAX_NUM_EXAMPLES    = 16
NUM_DISTINCT_TASKS  = 64
NOISE_VARIANCE      = .25

DATA = RegressionSequenceDistribution(
    task_distribution=DiscreteTaskDistribution(
        num_tasks=NUM_DISTINCT_TASKS,
        task_size=INPUT_DIMENSION,
        device=DEVICE,
    ),
    noise_variance=NOISE_VARIANCE,
)


# # MODEL SETTINGS

MODEL = InContextRegressionTransformer(
    task_size=INPUT_DIMENSION,
    max_examples=MAX_NUM_EXAMPLES,
    # beginner specs
    embed_size=64,
    mlp_size=64,
    num_heads=2,
    num_layers=2,
    # # specs from the paper
    # embed_size=128,
    # mlp_size=128,
    # num_heads=2,
    # num_layers=8,
    device=DEVICE,
)


# # TRAINING SETTINGS

BATCH_SIZE         = 256
# NUM_TRAINING_STEPS = 512
# NUM_TRAINING_STEPS = 65_536
NUM_TRAINING_STEPS = 524_288 # for the paper (500k)
LEARNING_RATE      = 1e-3


def LOSS(ys_true, ys_pred, axis=None):
    return (ys_true - ys_pred).square().mean(axis=axis)


# # EVALUATION

EVALUATION_BATCH_SIZE = 2048


@torch.no_grad()
def EVAL(step):
    report = ""
    def log(s):
        nonlocal report
        report += f"{s}\n"

    log(f"evaluation at step {step}:")

    log(f" generating new batch of {EVALUATION_BATCH_SIZE} from T_pretrain...")
    xs, ys = DATA.get_batch(
        num_examples=MAX_NUM_EXAMPLES,
        batch_size=EVALUATION_BATCH_SIZE,
    )
    
    log(" computing model and baseline predictions...")
    preds = {
        'model': MODEL(xs, ys),
        'dmmse': dmmse_predictor(
            xs,
            ys,
            prior=DATA.task_distribution,
            noise_variance=DATA.noise_variance,
        ),
        'ridge': ridge_predictor(
            xs,
            ys,
            noise_variance=DATA.noise_variance,
        ),
    }

    log(" average MSE loss for model and baselines:")
    # compute model and baseline outputs
    for m, ys_ in preds.items():
        log(f"  {m} mse loss = {LOSS(ys, ys_):.2f}")

    log(" MSE loss curves per-prediction, i.e., over the context:")
    for m, ys_ in preds.items():
        losses = LOSS(ys, ys_, axis=(0,2))
        string = ' '.join(f'{l:.2f}' for l in losses)
        log(f"  {m} mse loss curve = [ {string} ]")

    log(" average squared distance between model and baseline predictions:")
    log(f"  delta dMMSE = {LOSS(preds['dmmse'], preds['model']):.2f}")
    log(f"  delta ridge = {LOSS(preds['ridge'], preds['model']):.2f}")
    
    # TODO: also evaluate on gaussian task prior?

    return report



# # TRAINING LOOP

if __name__ == "__main__":

    # define optimizer
    optimizer = torch.optim.Adam(MODEL.parameters(), lr=LEARNING_RATE)

    # training loop
    for step in tqdm.trange(NUM_TRAINING_STEPS):
        # data generation and forward pass
        xs, ys = DATA.get_batch(
            num_examples=MAX_NUM_EXAMPLES,
            batch_size=BATCH_SIZE,
        )
        ys_pred = MODEL(xs, ys)
        loss = LOSS(ys_true=ys, ys_pred=ys_pred)
        # backward pass and gradient step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        # periodic diagnostic evaluation
        if NUM_TRAINING_STEPS <= 256 \
        or step % (NUM_TRAINING_STEPS // 256) == 0 \
        or step == NUM_TRAINING_STEPS - 1:
            tqdm.tqdm.write(EVAL(step))

