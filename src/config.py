from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
ANALYSIS_DIR = RESULTS_DIR / "analysis"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"

DEFAULT_RAW_DATA = RAW_DATA_DIR / "letras_generos_balanceado_sem_funk.csv"
DEFAULT_PROCESSED_DATA = PROCESSED_DATA_DIR / "letras_processadas_balanceado_sem_funk.csv"
DEFAULT_RANDOM_STATE = 42
DEFAULT_TEST_SIZE = 0.2


def ensure_project_dirs() -> None:
    for path in [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        MODELS_DIR,
        ANALYSIS_DIR,
        FIGURES_DIR,
        METRICS_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)
