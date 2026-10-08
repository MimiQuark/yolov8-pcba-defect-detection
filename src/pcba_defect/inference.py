from __future__ import annotations

import csv
import json
import logging
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .constants import CLASS_NAMES, CLASS_NAMES_ZH, DEFAULT_CONF, DEFAULT_DEVICE, DEFAULT_IMGSZ, DEFAULT_IOU
from .utils import ensure_dir, write_json

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class DetectionRecord:
    source: str
    class_id: int
    class_name: str
    class_name_zh: str
    confidence: float
    xmin: float
    ymin: float
    xmax: float
    ymax: float


def aggregate_counts(records: Iterable[DetectionRecord]) -> dict[str, int]:
    counts = {name: 0 for name in CLASS_NAMES}
    for record in records:
        counts[record.class_name] = counts.get(record.class_name, 0) + 1
    return counts


def decide_status(records: Iterable[DetectionRecord]) -> str:
    return "不合格" if any(True for _ in records) else "合格"


def records_from_result(result: Any, source: str | None = None) -> list[DetectionRecord]:
    records: list[DetectionRecord] = []
    boxes = getattr(result, "boxes", None)
    names = getattr(result, "names", {}) or {}
    result_source = source or Path(str(getattr(result, "path", "input"))).name
    if boxes is None or len(boxes) == 0:
        return records
    xyxy = boxes.xyxy.detach().cpu().numpy() if hasattr(boxes.xyxy, "detach") else np.asarray(boxes.xyxy)
    confs = boxes.conf.detach().cpu().numpy() if hasattr(boxes.conf, "detach") else np.asarray(boxes.conf)
    classes = boxes.cls.detach().cpu().numpy() if hasattr(boxes.cls, "detach") else np.asarray(boxes.cls)
    for box, confidence, class_value in zip(xyxy, confs, classes, strict=False):
        class_id = int(class_value)
        fallback = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else str(class_id)
        class_name = str(names.get(class_id, fallback))
        records.append(
            DetectionRecord(
                source=str(result_source),
                class_id=class_id,
                class_name=class_name,
                class_name_zh=CLASS_NAMES_ZH.get(class_name, class_name),
                confidence=float(confidence),
                xmin=float(box[0]),
                ymin=float(box[1]),
                xmax=float(box[2]),
                ymax=float(box[3]),
            )
        )
    return records


def write_records_csv(records: Iterable[DetectionRecord], output_path: str | Path) -> Path:
    rows = list(records)
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    fieldnames = [
        "source",
        "class_id",
        "class_name",
        "class_name_zh",
        "confidence",
        "xmin",
        "ymin",
        "xmax",
        "ymax",
    ]
    with output_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    return output_path


def _summary(records: list[DetectionRecord]) -> dict[str, Any]:
    counts = aggregate_counts(records)
    return {
        "status": decide_status(records),
        "detection_count": len(records),
        "class_counts": counts,
        "detections": [asdict(record) for record in records],
    }


def predict_source(
    model_path: str | Path,
    source: str | Path,
    output_dir: str | Path,
    conf: float = DEFAULT_CONF,
    iou: float = DEFAULT_IOU,
    imgsz: int = DEFAULT_IMGSZ,
    device: str = DEFAULT_DEVICE,
    save_txt: bool = False,
    save_csv: bool = True,
    show: bool = False,
) -> dict[str, Any]:
    from ultralytics import YOLO

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = ensure_dir(Path(output_dir).resolve() / timestamp)
    model = YOLO(str(model_path))
    result_stream = model.predict(
        source=str(source),
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        device=device,
        stream=True,
        save=True,
        save_txt=save_txt,
        save_conf=save_txt,
        show=show,
        project=str(run_dir),
        name="annotated",
        exist_ok=True,
        verbose=False,
    )

    by_source: dict[str, list[DetectionRecord]] = {}
    for index, result in enumerate(result_stream):
        path_value = getattr(result, "path", f"frame_{index:06d}")
        source_name = Path(str(path_value)).name
        records = records_from_result(result, source=source_name)
        by_source.setdefault(source_name, []).extend(records)

    all_records = [record for records in by_source.values() for record in records]
    per_source = {name: _summary(records) for name, records in by_source.items()}
    overall = _summary(all_records)
    csv_path = write_records_csv(all_records, run_dir / "detections.csv") if save_csv else None
    payload = {
        "model": str(Path(model_path).resolve()),
        "source": str(source),
        "conf": conf,
        "iou": iou,
        "imgsz": imgsz,
        "overall": overall,
        "per_source": per_source,
        "artifacts_dir": str(run_dir),
        "annotated_dir": str(run_dir / "annotated"),
        "csv": str(csv_path) if csv_path else None,
    }
    write_json(run_dir / "summary.json", payload)
    LOGGER.info("推理完成: %s", json.dumps(overall, ensure_ascii=False))
    return payload


def predict_image_array(
    model: Any,
    image: np.ndarray | None,
    output_dir: str | Path,
    conf: float = DEFAULT_CONF,
    iou: float = DEFAULT_IOU,
    imgsz: int = DEFAULT_IMGSZ,
    device: str = DEFAULT_DEVICE,
) -> tuple[np.ndarray | None, dict[str, Any], str | None]:
    if image is None:
        return None, {"error": "未提供图片"}, None
    image_bgr = image[:, :, ::-1].copy() if image.ndim == 3 and image.shape[2] == 3 else image
    result = model.predict(
        source=image_bgr,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        device=device,
        verbose=False,
    )[0]
    records = records_from_result(result, source="web_input")
    rendered_bgr = result.plot()
    rendered_rgb = rendered_bgr[:, :, ::-1].copy()
    payload = _summary(records)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    target = Path(output_dir).resolve() / "web" / timestamp / "detections.csv"
    csv_path = write_records_csv(records, target)
    payload["csv"] = str(csv_path)
    return rendered_rgb, payload, str(csv_path)
