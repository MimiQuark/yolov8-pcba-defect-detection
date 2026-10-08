from __future__ import annotations

import argparse
import json
from typing import Any

from .analysis import analyze_dataset
from .constants import (
    DEFAULT_ANALYSIS_DIR,
    DEFAULT_CONF,
    DEFAULT_DATASET_DIR,
    DEFAULT_DEVICE,
    DEFAULT_IMGSZ,
    DEFAULT_IOU,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SOURCE_ARCHIVE,
    PROJECT_ROOT,
)
from .dataset import prepare_dataset
from .training import evaluate_model, export_onnx, train_model
from .utils import configure_logging, find_best_model, resolve_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pcba",
        description="YOLOv8 PCBA 五类缺陷检测项目",
    )
    parser.add_argument("--verbose", action="store_true", help="输出调试日志")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="将 XML 数据集转换为 YOLO 格式")
    prepare.add_argument("--archive", default=str(DEFAULT_SOURCE_ARCHIVE), help="原始 ZIP 数据集")
    prepare.add_argument("--output", default=str(DEFAULT_DATASET_DIR), help="YOLO 数据集输出目录")
    prepare.add_argument("--lighting", choices=["auto", "clahe", "off"], default="auto")
    prepare.add_argument("--overwrite", action="store_true")

    analyze = subparsers.add_parser("analyze", help="生成数据统计和可视化")
    analyze.add_argument("--dataset", default=str(DEFAULT_DATASET_DIR))
    analyze.add_argument("--output", default=str(DEFAULT_ANALYSIS_DIR))

    train = subparsers.add_parser("train", help="训练 YOLOv8 模型")
    train.add_argument("--config", required=True, help="训练 YAML 配置")
    train.add_argument("--epochs", type=int)
    train.add_argument("--imgsz", type=int)
    train.add_argument("--batch", type=int)
    train.add_argument("--device", default=None)
    train.add_argument("--name", default=None)

    evaluate = subparsers.add_parser("eval", help="在指定划分上评估模型")
    evaluate.add_argument("--model", default=None)
    evaluate.add_argument("--data", default=str(DEFAULT_DATASET_DIR / "dataset.yaml"))
    evaluate.add_argument("--split", choices=["train", "val", "test"], default="test")
    evaluate.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    evaluate.add_argument("--batch", type=int, default=2)
    evaluate.add_argument("--device", default=DEFAULT_DEVICE)
    evaluate.add_argument("--output", default=str(PROJECT_ROOT / "runs" / "eval"))
    evaluate.add_argument("--name", default=None)

    predict = subparsers.add_parser("predict", help="对图片、目录、视频或摄像头执行推理")
    predict.add_argument("--model", default=None)
    predict.add_argument("--source", required=True)
    predict.add_argument("--output", default=str(DEFAULT_OUTPUT_DIR / "predict"))
    predict.add_argument("--conf", type=float, default=DEFAULT_CONF)
    predict.add_argument("--iou", type=float, default=DEFAULT_IOU)
    predict.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    predict.add_argument("--device", default=DEFAULT_DEVICE)
    predict.add_argument("--save-txt", action="store_true")
    predict.add_argument("--show", action="store_true")
    predict.add_argument("--no-save-csv", action="store_true")

    export = subparsers.add_parser("export", help="导出 ONNX 模型")
    export.add_argument("--model", default=None)
    export.add_argument("--output", default=str(PROJECT_ROOT / "models" / "pcba_yolov8n.onnx"))
    export.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    export.add_argument("--opset", type=int, default=17)
    export.add_argument("--device", default=DEFAULT_DEVICE)

    app = subparsers.add_parser("app", help="启动 Gradio 网页演示")
    app.add_argument("--model", default=None)
    app.add_argument("--host", default="127.0.0.1")
    app.add_argument("--port", type=int, default=7860)
    app.add_argument("--share", action="store_true")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.verbose)

    if args.command == "prepare":
        result = prepare_dataset(
            archive_path=resolve_path(args.archive),
            output_dir=resolve_path(args.output),
            lighting=args.lighting,
            overwrite=args.overwrite,
        )
    elif args.command == "analyze":
        result = analyze_dataset(
            dataset_dir=resolve_path(args.dataset),
            output_dir=resolve_path(args.output),
        )
    elif args.command == "train":
        overrides: dict[str, Any] = {
            "epochs": args.epochs,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "device": args.device,
            "name": args.name,
        }
        model_path = train_model(args.config, overrides)
        result = {"best_model": str(model_path)}
    elif args.command == "eval":
        model_path = find_best_model(args.model)
        result = evaluate_model(
            model_path=model_path,
            data_yaml=resolve_path(args.data),
            output_dir=resolve_path(args.output),
            split=args.split,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            name=args.name,
        )
    elif args.command == "predict":
        from .inference import predict_source

        model_path = find_best_model(args.model)
        result = predict_source(
            model_path=model_path,
            source=args.source,
            output_dir=resolve_path(args.output),
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=args.device,
            save_txt=args.save_txt,
            save_csv=not args.no_save_csv,
            show=args.show,
        )
    elif args.command == "export":
        model_path = find_best_model(args.model)
        result = export_onnx(
            model_path=model_path,
            output_path=resolve_path(args.output),
            imgsz=args.imgsz,
            opset=args.opset,
            device=args.device,
        )
    elif args.command == "app":
        from .app import launch_app

        launch_app(
            model_path=args.model,
            host=args.host,
            port=args.port,
            share=args.share,
        )
        result = {"status": "stopped"}
    else:
        parser.error(f"未知命令: {args.command}")
        return 2

    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
