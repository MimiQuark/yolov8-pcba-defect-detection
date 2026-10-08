from __future__ import annotations

from pcba_defect.inference import DetectionRecord, aggregate_counts, decide_status


def test_status_and_counts() -> None:
    record = DetectionRecord(
        source="sample.jpg",
        class_id=1,
        class_name="short",
        class_name_zh="短路",
        confidence=0.9,
        xmin=1,
        ymin=2,
        xmax=3,
        ymax=4,
    )
    assert decide_status([]) == "合格"
    assert decide_status([record]) == "不合格"
    assert aggregate_counts([record])["short"] == 1
