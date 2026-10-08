# 已发布实验结果

本目录保存无需重新训练即可查看或复现实验结论的文件。原始数据集、训练缓存和大量中间图片不进入 Git。

## 模型

- `models/pcba_yolov8n_best.pt`：Ultralytics YOLOv8n 最佳 PyTorch 权重。
- `models/pcba_yolov8n.onnx`：固定 960 输入的 FP32 ONNX 模型，已通过 ONNX Runtime CPU 加载验证。

## 测试集指标

- Precision：94.18%
- Recall：77.42%
- mAP50：87.94%
- mAP50-95：53.13%

逐类别结果见 `results/evaluation/per_class_metrics.csv`，完整原始指标见 `metrics.json`。

## 报告与图表

- `reports/`：Word 和 PDF 实验报告。
- `results/analysis/`：数据分布和标注样例。
- `results/training/`：训练配置、逐轮日志和训练曲线。
- `results/evaluation/`：测试集指标和归一化混淆矩阵。
- `results/web_demo.png`：Gradio 网页实际检测截图。
- `results/prediction_351.png`：测试样图 351 的最终检测结果。
