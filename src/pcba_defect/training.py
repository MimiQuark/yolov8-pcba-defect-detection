from __future__ import annotations

import csv
import json
import logging
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .constants import CLASS_NAMES, DEFAULT_DEVICE, DEFAULT_IMGSZ, PROJECT_ROOT
from .utils import ensure_dir, resolve_path, write_json

LOGGER = logging.getLogger(__name__)


def _to_builtin(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(item) for item in value]
    return value


def load_training_config(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"训练配置不存在: {config_path}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(config, dict):
        raise ValueError(f"训练配置必须是 YAML 对象: {config_path}")

    config["config_path"] = str(config_path)
    if config.get("data"):
        config["data"] = str(resolve_path(config["data"]))
    if config.get("project"):
        config["project"] = str(resolve_path(config["project"]))
    return config


def train_model(config_path: str | Path, overrides: dict[str, Any] | None = None) -> Path:
    from ultralytics import YOLO

    config = load_training_config(config_path)
    overrides = {key: value for key, value in (overrides or {}).items() if value is not None}
    config.update(overrides)

    data_path = Path(config["data"]).resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"YOLO 数据配置不存在: {data_path}")

    model_name = config.pop("model", "yolov8n.pt")
    config.pop("config_path", None)
    config["data"] = str(data_path)
    config["project"] = str(ensure_dir(config.get("project", PROJECT_ROOT / "runs" / "train")))
    config.setdefault("exist_ok", False)
    config.setdefault("plots", True)
    config.setdefault("save", True)
    config.setdefault("val", True)
    config.setdefault("device", DEFAULT_DEVICE)
    config.setdefault("imgsz", DEFAULT_IMGSZ)

    LOGGER.info("开始训练: model=%s, data=%s, imgsz=%s, epochs=%s", model_name, data_path, config.get("imgsz"), config.get("epochs"))
    model = YOLO(model_name)
    model.train(**config)
    train_dir = Path(model.trainer.save_dir).resolve()
    weights_dir = train_dir / "weights"
    best = weights_dir / "best.pt"
    if not best.exists():
        raise RuntimeError(f"训练结束但未找到 best.pt: {best}")
    destination = ensure_dir(PROJECT_ROOT / "models") / "pcba_yolov8n_best.pt"
    shutil.copy2(best, destination)
    write_json(
        train_dir / "training_config.json",
        {
            "model": model_name,
            "resolved_args": _to_builtin(config),
            "best_weights": str(best),
            "copied_weights": str(destination),
        },
    )
    LOGGER.info("训练完成: %s", destination)
    return destination


def _class_names_from_result(result: Any) -> list[str]:
    names = getattr(result, "names", None) or {index: name for index, name in enumerate(CLASS_NAMES)}
    return [str(names.get(index, names.get(str(index), f"class_{index}"))) for index in range(len(names))]


def evaluate_model(
    model_path: str | Path,
    data_yaml: str | Path,
    output_dir: str | Path,
    split: str = "test",
    imgsz: int = DEFAULT_IMGSZ,
    batch: int = 2,
    device: str = DEFAULT_DEVICE,
    name: str | None = None,
) -> dict[str, Any]:
    from ultralytics import YOLO

    model_path = Path(model_path).resolve()
    data_yaml = Path(data_yaml).resolve()
    output_dir = ensure_dir(output_dir)
    run_name = name or f"{split}_{imgsz}"
    model = YOLO(str(model_path))
    metrics = model.val(
        data=str(data_yaml),
        split=split,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=str(output_dir),
        name=run_name,
        plots=True,
        save_json=False,
        verbose=True,
    )
    save_dir = Path(metrics.save_dir).resolve()
    names = _class_names_from_result(metrics)
    box = metrics.box
    per_class = []
    for index, class_name in enumerate(names):
        per_class.append(
            {
                "class_id": index,
                "class_name": class_name,
                "precision": float(box.p[index]) if index < len(box.p) else 0.0,
                "recall": float(box.r[index]) if index < len(box.r) else 0.0,
                "mAP50": float(box.ap50[index]) if index < len(box.ap50) else 0.0,
                "mAP50_95": float(box.ap[index]) if index < len(box.ap) else 0.0,
            }
        )

    payload = {
        "model": str(model_path),
        "data": str(data_yaml),
        "split": split,
        "imgsz": imgsz,
        "batch": batch,
        "device": device,
        "overall": _to_builtin(metrics.results_dict),
        "per_class": per_class,
        "speed_ms_per_image": _to_builtin(metrics.speed),
        "save_dir": str(save_dir),
    }
    write_json(save_dir / "metrics.json", payload)
    with (save_dir / "per_class_metrics.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_class[0]))
        writer.writeheader()
        writer.writerows(per_class)

    confusion = getattr(getattr(metrics, "confusion_matrix", None), "matrix", None)
    if confusion is not None:
        np.savetxt(save_dir / "confusion_matrix.csv", np.asarray(confusion), delimiter=",", fmt="%.6f")
    LOGGER.info("评估完成: %s", json.dumps(payload["overall"], ensure_ascii=False))
    return payload


def export_onnx(
    model_path: str | Path,
    output_path: str | Path,
    imgsz: int = DEFAULT_IMGSZ,
    opset: int = 17,
    device: str = DEFAULT_DEVICE,
) -> dict[str, Any]:
    import onnx
    import onnxruntime as ort
    from ultralytics import YOLO

    model_path = Path(model_path).resolve()
    output_path = Path(output_path).resolve()
    ensure_dir(output_path.parent)
    model = YOLO(str(model_path))
    exported = Path(
        model.export(
            format="onnx",
            imgsz=imgsz,
            opset=opset,
            dynamic=False,
            simplify=False,
            device=device,
        )
    ).resolve()
    if exported != output_path:
        shutil.copy2(exported, output_path)

    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    payload = {
        "model_path": str(model_path),
        "onnx_path": str(output_path),
        "imgsz": imgsz,
        "opset": opset,
        "inputs": [{"name": item.name, "shape": item.shape, "type": item.type} for item in session.get_inputs()],
        "outputs": [{"name": item.name, "shape": item.shape, "type": item.type} for item in session.get_outputs()],
    }
    write_json(output_path.with_suffix(".json"), payload)
    LOGGER.info("ONNX 导出与加载验证完成: %s", output_path)
    return payload
