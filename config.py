import os
from dotenv import load_dotenv

load_dotenv(dotenv_path="local.env")

WANDB_API_KEY = os.getenv("WANDB_API_KEY")
RANDOM_SEED = 321
PROJECT_NAME = "zbmath_recsys"
