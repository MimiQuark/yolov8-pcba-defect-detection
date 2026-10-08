from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any

from .constants import PROJECT_ROOT


def ensure_dir(path: str | Path) -> Path:
    path = Path(path).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_path(path: str | Path | None, default: str | Path | None = None) -> Path | None:
    if path is None or str(path) == "":
        return Path(default).expanduser().resolve() if default is not None else None
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate.resolve()


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, data: Any) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def find_best_model(explicit: str | Path | None = None) -> Path:
    if explicit is not None and str(explicit) != "":
        path = resolve_path(explicit)
        if path and path.exists():
            return path
        raise FileNotFoundError(f"模型文件不存在: {explicit}")

    candidates = [
        PROJECT_ROOT / "models" / "pcba_yolov8n_best.pt",
        PROJECT_ROOT / "runs" / "train" / "pcba_yolov8n_960" / "weights" / "best.pt",
    ]
    candidates.extend(sorted((PROJECT_ROOT / "runs" / "train").glob("**/weights/best.pt"), key=lambda p: p.stat().st_mtime, reverse=True))
    for path in candidates:
        if path.exists():
            return path.resolve()
    raise FileNotFoundError("未找到已训练模型，请先运行 `pcba train` 或通过 --model 指定模型。")


def safe_replace_dir(target: str | Path, allowed_parent: str | Path) -> None:
    target = Path(target).resolve()
    allowed_parent = Path(allowed_parent).resolve()
    if target == allowed_parent or allowed_parent not in target.parents:
        raise ValueError(f"拒绝删除不安全路径: {target}")
    if target.exists():
        shutil.rmtree(target)


def get_env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}
