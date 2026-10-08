from __future__ import annotations

from pathlib import Path
from typing import Any

from .constants import DEFAULT_CONF, DEFAULT_DEVICE, DEFAULT_IMGSZ, DEFAULT_IOU, PROJECT_ROOT
from .inference import predict_image_array, predict_source
from .utils import ensure_dir, find_best_model

_MODEL: Any | None = None
_MODEL_PATH: Path | None = None


def _get_model(model_path: str | Path | None = None) -> Any:
    global _MODEL, _MODEL_PATH
    from ultralytics import YOLO

    resolved = find_best_model(model_path)
    if _MODEL is None or _MODEL_PATH != resolved:
        _MODEL = YOLO(str(resolved))
        _MODEL_PATH = resolved
    return _MODEL


def _format_summary(summary: dict[str, Any]) -> str:
    if "error" in summary:
        return f"### 检测失败\n\n{summary['error']}"
    status = summary.get("status", "未知")
    color = "red" if status == "不合格" else "green"
    counts = summary.get("class_counts", {})
    lines = [
        f"### 检测结论：<span style='color:{color}'>{status}</span>",
        "",
        f"- 检测目标数：{summary.get('detection_count', 0)}",
    ]
    for name, count in counts.items():
        if count:
            lines.append(f"- {name}：{count}")
    return "\n".join(lines)


def _detections_table(summary: dict[str, Any]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for item in summary.get("detections", []):
        rows.append(
            [
                item.get("class_name_zh", item.get("class_name")),
                item.get("class_name"),
                round(float(item.get("confidence", 0.0)), 4),
                round(float(item.get("xmin", 0.0)), 2),
                round(float(item.get("ymin", 0.0)), 2),
                round(float(item.get("xmax", 0.0)), 2),
                round(float(item.get("ymax", 0.0)), 2),
            ]
        )
    return rows


def create_app(model_path: str | Path | None = None) -> Any:
    import gradio as gr

    resolved_model = find_best_model(model_path)
    _get_model(resolved_model)
    output_dir = ensure_dir(PROJECT_ROOT / "outputs")

    def detect_image(
        image: Any,
        conf: float,
        iou: float,
        imgsz: int,
    ) -> tuple[Any, str, list[list[Any]], str | None]:
        model = _get_model(resolved_model)
        rendered, payload, csv_path = predict_image_array(
            model,
            image,
            output_dir=output_dir,
            conf=float(conf),
            iou=float(iou),
            imgsz=int(imgsz),
            device=DEFAULT_DEVICE,
        )
        return rendered, _format_summary(payload), _detections_table(payload), csv_path

    def detect_video(
        video_path: str | None,
        conf: float,
        iou: float,
        imgsz: int,
    ) -> tuple[Any, str, list[list[Any]], str | None]:
        if not video_path:
            return None, _format_summary({"error": "未提供视频"}), [], None
        payload = predict_source(
            resolved_model,
            video_path,
            output_dir=output_dir / "video",
            conf=float(conf),
            iou=float(iou),
            imgsz=int(imgsz),
            device=DEFAULT_DEVICE,
            save_txt=False,
            save_csv=True,
        )
        annotated_dir = Path(payload["annotated_dir"])
        candidates = sorted(
            [path for path in annotated_dir.glob("**/*") if path.is_file()],
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        rendered = str(candidates[0]) if candidates else video_path
        overall = payload["overall"]
        return rendered, _format_summary(overall), _detections_table(overall), payload.get("csv")

    with gr.Blocks(title="PCBA 五类缺陷检测系统") as demo:
        gr.Markdown(
            "# PCBA 五类缺陷检测系统\n"
            "基于 YOLOv8，支持图片、视频和摄像头输入。检测到任一缺陷时判定为“不合格”。"
        )
        with gr.Row():
            conf = gr.Slider(0.05, 0.95, value=DEFAULT_CONF, step=0.05, label="置信度阈值")
            iou = gr.Slider(0.1, 0.9, value=DEFAULT_IOU, step=0.05, label="IoU 阈值")
            imgsz = gr.Dropdown([640, 800, 960, 1280], value=DEFAULT_IMGSZ, label="推理尺寸")

        with gr.Tab("图片与摄像头"):
            image_input = gr.Image(type="numpy", sources=["upload", "webcam"], label="上传图片或拍摄")
            image_button = gr.Button("开始检测", variant="primary")
            image_output = gr.Image(label="检测结果")
            image_summary = gr.Markdown()
            image_table = gr.Dataframe(
                headers=["类别", "英文名", "置信度", "xmin", "ymin", "xmax", "ymax"],
                label="检测明细",
            )
            image_csv = gr.File(label="下载检测 CSV")
            image_button.click(
                detect_image,
                inputs=[image_input, conf, iou, imgsz],
                outputs=[image_output, image_summary, image_table, image_csv],
            )

        with gr.Tab("视频"):
            video_input = gr.Video(label="上传视频")
            video_button = gr.Button("开始检测", variant="primary")
            video_output = gr.Video(label="标注视频")
            video_summary = gr.Markdown()
            video_table = gr.Dataframe(
                headers=["类别", "英文名", "置信度", "xmin", "ymin", "xmax", "ymax"],
                label="全视频检测明细",
            )
            video_csv = gr.File(label="下载检测 CSV")
            video_button.click(
                detect_video,
                inputs=[video_input, conf, iou, imgsz],
                outputs=[video_output, video_summary, video_table, video_csv],
            )
        gr.Markdown(
            f"模型路径：`{resolved_model}`\n\n"
            "CPU 推理时 960 尺寸可能需要数秒至数十秒；现场演示可先使用 640。"
        )
    return demo


def launch_app(
    model_path: str | Path | None = None,
    host: str = "127.0.0.1",
    port: int = 7860,
    share: bool = False,
) -> None:
    import gradio as gr

    demo = create_app(model_path)
    demo.launch(
        server_name=host,
        server_port=port,
        share=share,
        show_error=True,
        theme=gr.themes.Soft(),
    )
