from pathlib import Path

import torch

REPO_PATH = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_PATH / "data"
LOGS_PATH = REPO_PATH / "logs"
RUNS_PATH = REPO_PATH / "runs"

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
FLOAT_DTYPE = torch.float32
