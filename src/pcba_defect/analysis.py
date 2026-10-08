from __future__ import annotations

import csv
import logging
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .constants import CLASS_NAMES, CLASS_NAMES_ZH, SPLITS
from .dataset import yolo_to_xbox
from .utils import ensure_dir, write_json

LOGGER = logging.getLogger(__name__)


def _label_rows(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    rows: list[tuple[int, float, float, float, float]] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        rows.append((int(parts[0]), *(float(value) for value in parts[1:])))
    return rows


def _draw_samples(
    samples: list[tuple[Path, list[tuple[int, float, float, float, float]], tuple[int, int]]],
    output: Path,
) -> None:
    tiles: list[Image.Image] = []
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except OSError:
        font = ImageFont.load_default()
    for image_path, rows, size in samples:
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        for class_id, xc, yc, bw, bh in rows:
            x1, y1, x2, y2 = yolo_to_xbox(xc, yc, bw, bh, size[0], size[1])
            draw.rectangle((x1, y1, x2, y2), outline="red", width=8)
            text = CLASS_NAMES[class_id]
            label_origin = (x1, max(0, y1 - 28))
            bbox = draw.textbbox(label_origin, text, font=font, stroke_width=2)
            draw.rectangle(bbox, fill=(0, 0, 0))
            draw.text(label_origin, text, fill="yellow", font=font, stroke_width=1, stroke_fill="black")
        image.thumbnail((700, 580))
        tiles.append(image)

    if not tiles:
        return
    cell_w = max(tile.width for tile in tiles)
    cell_h = max(tile.height for tile in tiles)
    grid = Image.new("RGB", (cell_w * 3, cell_h * 2), "white")
    for index, tile in enumerate(tiles[:6]):
        x = (index % 3) * cell_w
        y = (index // 3) * cell_h
        grid.paste(tile, (x, y))
    ensure_dir(output.parent)
    grid.save(output, quality=92)


def analyze_dataset(dataset_dir: str | Path, output_dir: str | Path) -> dict[str, Any]:
    dataset_dir = Path(dataset_dir).resolve()
    output_dir = ensure_dir(output_dir)
    if not (dataset_dir / "dataset.yaml").exists():
        raise FileNotFoundError(f"未找到 YOLO 数据集配置: {dataset_dir / 'dataset.yaml'}")

    summary: dict[str, Any] = {
        "dataset_dir": str(dataset_dir),
        "class_names": list(CLASS_NAMES),
        "class_names_zh": CLASS_NAMES_ZH,
        "splits": {},
    }
    all_brightness: list[float] = []
    class_rows: list[dict[str, Any]] = []
    size_rows: list[dict[str, Any]] = []
    samples: list[tuple[Path, list[tuple[int, float, float, float, float]], tuple[int, int]]] = []

    for split in SPLITS:
        image_dir = dataset_dir / "images" / split
        label_dir = dataset_dir / "labels" / split
        images = sorted(path for path in image_dir.iterdir() if path.is_file())
        object_counts: Counter[str] = Counter()
        image_counts: Counter[str] = Counter()
        brightness_values: list[float] = []
        split_objects = 0
        for image_path in images:
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                raise FileNotFoundError(f"缺少标签文件: {label_path}")
            rows = _label_rows(label_path)
            with Image.open(image_path) as image:
                size = image.size
                gray = np.asarray(image.convert("L"), dtype=np.uint8)
            brightness_values.append(float(gray.mean()))
            present: set[str] = set()
            for class_id, _xc, _yc, bw, bh in rows:
                name = CLASS_NAMES[class_id]
                object_counts[name] += 1
                present.add(name)
                size_rows.append(
                    {
                        "split": split,
                        "image": image_path.name,
                        "class": name,
                        "width_px": round(bw * size[0], 3),
                        "height_px": round(bh * size[1], 3),
                        "area_ratio": round(bw * bh, 8),
                    }
                )
            for name in present:
                image_counts[name] += 1
            split_objects += len(rows)
            if len(samples) < 6 and len(rows) >= 6:
                samples.append((image_path, rows, size))

        all_brightness.extend(brightness_values)
        summary["splits"][split] = {
            "images": len(images),
            "objects": split_objects,
            "class_objects": {name: object_counts[name] for name in CLASS_NAMES},
            "class_images": {name: image_counts[name] for name in CLASS_NAMES},
            "brightness": {
                "min": round(min(brightness_values), 3),
                "mean": round(statistics.mean(brightness_values), 3),
                "max": round(max(brightness_values), 3),
            },
            "mean_objects_per_image": round(split_objects / max(len(images), 1), 3),
        }
        for name in CLASS_NAMES:
            class_rows.append(
                {
                    "split": split,
                    "class": name,
                    "class_zh": CLASS_NAMES_ZH[name],
                    "images": image_counts[name],
                    "objects": object_counts[name],
                }
            )

    summary["totals"] = {
        "images": sum(summary["splits"][split]["images"] for split in SPLITS),
        "objects": sum(summary["splits"][split]["objects"] for split in SPLITS),
        "class_objects": {
            name: sum(summary["splits"][split]["class_objects"][name] for split in SPLITS)
            for name in CLASS_NAMES
        },
    }
    summary["brightness"] = {
        "min": round(min(all_brightness), 3),
        "mean": round(statistics.mean(all_brightness), 3),
        "max": round(max(all_brightness), 3),
    }

    with (output_dir / "class_distribution.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(class_rows[0]))
        writer.writeheader()
        writer.writerows(class_rows)

    with (output_dir / "object_sizes.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(size_rows[0]))
        writer.writeheader()
        writer.writerows(size_rows)

    plt.figure(figsize=(10, 5))
    plt.hist(all_brightness, bins=30, color="#2574a9", edgecolor="white")
    plt.axvline(60, color="red", linestyle="--", label="low threshold")
    plt.axvline(200, color="orange", linestyle="--", label="high threshold")
    plt.title("PCBA Image Brightness Distribution")
    plt.xlabel("Mean grayscale")
    plt.ylabel("Images")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "brightness_hist.png", dpi=180)
    plt.close()

    x = np.arange(len(CLASS_NAMES))
    width = 0.25
    plt.figure(figsize=(11, 5))
    for offset, split in enumerate(SPLITS):
        values = [summary["splits"][split]["class_objects"][name] for name in CLASS_NAMES]
        plt.bar(x + (offset - 1) * width, values, width, label=split)
    plt.xticks(x, CLASS_NAMES, rotation=20)
    plt.title("PCBA Defect Object Distribution")
    plt.ylabel("Objects")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "class_distribution.png", dpi=180)
    plt.close()

    _draw_samples(samples, output_dir / "samples_annotated.jpg")
    write_json(output_dir / "summary.json", summary)
    LOGGER.info("数据分析完成: %s", output_dir)
    return summary

