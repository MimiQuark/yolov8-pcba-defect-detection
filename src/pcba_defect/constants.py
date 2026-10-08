from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CLASS_NAMES: tuple[str, ...] = (
    "open_solder",
    "short",
    "skewing",
    "solder_bridge",
    "tombstoning",
)

CLASS_TO_ID: dict[str, int] = {name: idx for idx, name in enumerate(CLASS_NAMES)}
ID_TO_NAME: dict[int, str] = {idx: name for name, idx in CLASS_TO_ID.items()}
CLASS_NAMES_ZH: dict[str, str] = {
    "open_solder": "开焊",
    "short": "短路",
    "skewing": "偏移",
    "solder_bridge": "连锡",
    "tombstoning": "立碑",
}

SPLITS: tuple[str, ...] = ("train", "val", "test")
IMAGE_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.45
DEFAULT_IMGSZ = 960
DEFAULT_DEVICE = "cpu"
DEFAULT_SEED = 42

DEFAULT_DATASET_DIR = PROJECT_ROOT / "data" / "pcba_yolo"
DEFAULT_ANALYSIS_DIR = PROJECT_ROOT / "data" / "analysis"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models"
DEFAULT_RUNS_DIR = PROJECT_ROOT / "runs"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "reports"
DEFAULT_SOURCE_ARCHIVE = Path(r"C:\Users\27800\Downloads\data\data139469\data_y.zip")
