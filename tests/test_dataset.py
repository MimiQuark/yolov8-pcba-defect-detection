from __future__ import annotations

from PIL import Image

from pcba_defect.dataset import apply_lighting, xbox_to_yolo, yolo_to_xbox


def test_xbox_yolo_roundtrip() -> None:
    width, height = 2448, 2048
    original = (524.0, 703.0, 583.0, 796.0)
    converted = xbox_to_yolo(*original, width, height)
    restored = yolo_to_xbox(*converted, width, height)
    assert max(abs(a - b) for a, b in zip(original, restored, strict=True)) <= 1.0


def test_auto_lighting_keeps_normal_image() -> None:
    image = Image.new("RGB", (32, 32), color=(120, 120, 120))
    processed, action = apply_lighting(image, "auto")
    assert processed is image
    assert action == "no_change"
