from __future__ import annotations

import io
import logging
import statistics
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from PIL import Image, ImageEnhance

from .constants import CLASS_NAMES, CLASS_TO_ID, IMAGE_EXTENSIONS, SPLITS
from .utils import ensure_dir, write_json

LOGGER = logging.getLogger(__name__)


def xbox_to_yolo(
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    if width <= 0 or height <= 0:
        raise ValueError(f"非法图像尺寸: {width}x{height}")
    if not (0 <= xmin < xmax <= width and 0 <= ymin < ymax <= height):
        raise ValueError(f"非法目标框: ({xmin}, {ymin}, {xmax}, {ymax}) / ({width}, {height})")
    box_w = xmax - xmin
    box_h = ymax - ymin
    return (
        (xmin + xmax) / 2 / width,
        (ymin + ymax) / 2 / height,
        box_w / width,
        box_h / height,
    )


def yolo_to_xbox(
    x_center: float,
    y_center: float,
    box_w: float,
    box_h: float,
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    xmin = (x_center - box_w / 2) * width
    ymin = (y_center - box_h / 2) * height
    xmax = (x_center + box_w / 2) * width
    ymax = (y_center + box_h / 2) * height
    return xmin, ymin, xmax, ymax


def mean_luminance(image: Image.Image) -> float:
    grayscale = np.asarray(image.convert("L"), dtype=np.uint8)
    return float(grayscale.mean())


def apply_lighting(
    image: Image.Image,
    mode: str = "auto",
    low: float = 60.0,
    high: float = 200.0,
) -> tuple[Image.Image, str]:
    mode = mode.lower()
    if mode not in {"auto", "clahe", "off"}:
        raise ValueError("lighting 必须是 auto、clahe 或 off")

    if mode == "off":
        return image, "off"

    brightness = mean_luminance(image)
    if mode == "auto":
        if low <= brightness <= high:
            return image, "no_change"
        factor = 128.0 / max(brightness, 1.0) if brightness < low else high / brightness
        factor = float(np.clip(factor, 0.5, 2.5))
        enhanced = ImageEnhance.Brightness(image).enhance(factor)
        return enhanced, f"brightness_{factor:.3f}"

    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    merged = cv2.merge((l_channel, a_channel, b_channel))
    result = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
    return Image.fromarray(result), "clahe"


def _iter_split_entries(zf: zipfile.ZipFile, split: str) -> tuple[dict[str, str], dict[str, str]]:
    names = zf.namelist()
    marker = f"/{split}/"
    image_entries: dict[str, str] = {}
    annotation_entries: dict[str, str] = {}

    for name in names:
        if name.endswith("/"):
            continue
        normalized = "/" + name.replace("\\", "/")
        if marker not in normalized:
            continue
        suffix = Path(name).suffix.lower()
        stem = Path(name).stem
        if suffix in IMAGE_EXTENSIONS and "/images/" in normalized:
            if stem in image_entries:
                raise ValueError(f"{split} 中图片编号重复: {stem}")
            image_entries[stem] = name
        elif suffix == ".xml" and "/annotations/" in normalized:
            if stem in annotation_entries:
                raise ValueError(f"{split} 中标注编号重复: {stem}")
            annotation_entries[stem] = name

    if set(image_entries) != set(annotation_entries):
        missing_annotations = sorted(set(image_entries) - set(annotation_entries))[:20]
        missing_images = sorted(set(annotation_entries) - set(image_entries))[:20]
        raise ValueError(
            f"{split} 图片与 XML 不匹配；缺 XML: {missing_annotations}；缺图片: {missing_images}"
        )
    return image_entries, annotation_entries


def prepare_dataset(
    archive_path: str | Path,
    output_dir: str | Path,
    lighting: str = "auto",
    overwrite: bool = False,
) -> dict[str, Any]:
    archive_path = Path(archive_path).resolve()
    output_dir = Path(output_dir).resolve()
    if not archive_path.exists():
        raise FileNotFoundError(f"数据集压缩包不存在: {archive_path}")

    dataset_yaml = output_dir / "dataset.yaml"
    if dataset_yaml.exists() and not overwrite:
        LOGGER.info("数据集已存在: %s；如需重新生成请添加 --overwrite", output_dir)

    for split in SPLITS:
        ensure_dir(output_dir / "images" / split)
        ensure_dir(output_dir / "labels" / split)

    image_counts: Counter[str] = Counter()
    object_counts: Counter[str] = Counter()
    class_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
    brightness_by_split: defaultdict[str, list[float]] = defaultdict(list)
    lighting_actions: Counter[str] = Counter()
    sizes: Counter[tuple[int, int]] = Counter()

    with zipfile.ZipFile(archive_path) as zf:
        for split in SPLITS:
            image_entries, annotation_entries = _iter_split_entries(zf, split)
            ordered_stems = sorted(image_entries, key=lambda value: int(value) if value.isdigit() else value)
            for index, stem in enumerate(ordered_stems, start=1):
                image_name = image_entries[stem]
                annotation_name = annotation_entries[stem]
                image_bytes = zf.read(image_name)
                xml_root = ET.fromstring(zf.read(annotation_name))

                size_node = xml_root.find("size")
                if size_node is None:
                    raise ValueError(f"{annotation_name} 缺少 <size>")
                width = int(float(size_node.findtext("width", "0")))
                height = int(float(size_node.findtext("height", "0")))

                with Image.open(io.BytesIO(image_bytes)) as source_image:
                    image = source_image.convert("RGB")
                if image.size != (width, height):
                    raise ValueError(
                        f"{annotation_name} 尺寸 {width}x{height} 与图片实际尺寸 {image.size[0]}x{image.size[1]} 不一致"
                    )

                brightness = mean_luminance(image)
                brightness_by_split[split].append(brightness)
                processed_image, action = apply_lighting(image, mode=lighting)
                lighting_actions[action] += 1
                sizes[(width, height)] += 1

                suffix = Path(image_name).suffix.lower()
                output_image = output_dir / "images" / split / f"{stem}{suffix}"
                if action == "no_change" or action == "off":
                    output_image.write_bytes(image_bytes)
                else:
                    processed_image.save(output_image, quality=95, subsampling=0)

                labels: list[str] = []
                for obj in xml_root.findall("object"):
                    class_name = (obj.findtext("name") or "").strip()
                    if class_name not in CLASS_TO_ID:
                        raise ValueError(f"{annotation_name} 含未知类别: {class_name}")
                    box = obj.find("bndbox")
                    if box is None:
                        raise ValueError(f"{annotation_name} 中 {class_name} 缺少 <bndbox>")
                    xmin = float(box.findtext("xmin", "0"))
                    ymin = float(box.findtext("ymin", "0"))
                    xmax = float(box.findtext("xmax", "0"))
                    ymax = float(box.findtext("ymax", "0"))
                    x_center, y_center, box_w, box_h = xbox_to_yolo(
                        xmin, ymin, xmax, ymax, width, height
                    )
                    labels.append(
                        f"{CLASS_TO_ID[class_name]} {x_center:.8f} {y_center:.8f} {box_w:.8f} {box_h:.8f}"
                    )
                    object_counts[split] += 1
                    class_counts[split][class_name] += 1

                if not labels:
                    raise ValueError(f"{annotation_name} 没有目标框")
                (output_dir / "labels" / split / f"{stem}.txt").write_text(
                    "\n".join(labels) + "\n", encoding="utf-8"
                )
                image_counts[split] += 1
                if index % 50 == 0:
                    LOGGER.info("%s 已处理 %d/%d", split, index, len(ordered_stems))

    yaml_data = {
        "path": output_dir.as_posix(),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {idx: name for idx, name in enumerate(CLASS_NAMES)},
    }
    dataset_yaml.write_text(
        yaml.safe_dump(yaml_data, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )

    split_summary: dict[str, Any] = {}
    for split in SPLITS:
        values = brightness_by_split[split]
        split_summary[split] = {
            "images": image_counts[split],
            "objects": object_counts[split],
            "class_counts": {name: class_counts[split][name] for name in CLASS_NAMES},
            "brightness": {
                "min": round(min(values), 3),
                "mean": round(statistics.mean(values), 3),
                "max": round(max(values), 3),
            },
        }

    summary = {
        "source_archive": str(archive_path),
        "output_dir": str(output_dir),
        "dataset_yaml": str(dataset_yaml),
        "class_mapping": {name: CLASS_TO_ID[name] for name in CLASS_NAMES},
        "splits": split_summary,
        "totals": {
            "images": sum(image_counts.values()),
            "objects": sum(object_counts.values()),
            "class_counts": {
                name: sum(class_counts[split][name] for split in SPLITS) for name in CLASS_NAMES
            },
            "sizes": {f"{w}x{h}": count for (w, h), count in sizes.items()},
        },
        "lighting_mode": lighting,
        "lighting_actions": dict(lighting_actions),
        "validation": {
            "image_annotation_pairs": sum(image_counts.values()),
            "unknown_classes": 0,
            "invalid_boxes": 0,
            "missing_labels": 0,
        },
    }
    write_json(output_dir / "dataset_summary.json", summary)
    write_json(output_dir / "preparation_report.json", summary)
    LOGGER.info(
        "数据准备完成：%d 张图片，%d 个目标框，dataset.yaml=%s",
        summary["totals"]["images"],
        summary["totals"]["objects"],
        dataset_yaml,
    )
    return summary
