import os
import wandb
import pytorch_lightning as pl

num_gpus = int(os.environ.get('SLURM_GPUS', 1))
print(num_gpus)
