"""Central paths resolved relative to the project root."""

import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
SERVER_INFO_PATH = os.path.join(DATA_DIR, "server_info.xlsx")
TRAIN_TASK_PATH = os.path.join(DATA_DIR, "task_parameters.xlsx")
EVALUATION_TASK_PATH = os.path.join(DATA_DIR, "task_parameters_evaluation.xlsx")


def ensure_dirs() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
