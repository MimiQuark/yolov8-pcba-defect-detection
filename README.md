# YOLOv8 PCBA 五类缺陷检测

本项目以 `data_y.zip` 中的 PCBA 图像和 Pascal VOC XML 标注为数据源，使用 Ultralytics YOLOv8 完成数据准备、分析、训练、测试、推理、ONNX 导出和 Gradio 网页演示。项目默认面向无 NVIDIA GPU 的 Windows CPU 环境。

## 类别

| ID | 英文类别 | 中文含义 |
|---:|---|---|
| 0 | open_solder | 开焊 |
| 1 | short | 短路 |
| 2 | skewing | 偏移 |
| 3 | solder_bridge | 连锡 |
| 4 | tombstoning | 立碑 |

只要检测到任一类别，业务结论即为“不合格”；未检测到目标则结论为“合格”。

## 数据准备

将原始 `data_y.zip` 放到 `data/source/data_y.zip`，或者通过 `pcba prepare --archive <路径>` 显式指定压缩包位置。原始数据不会提交到 Git。

## 数据概况

- 原始数据：600 张 JPG 图片、600 个 XML 标注、4552 个目标框。
- 原始划分：训练集 350 张、验证集 100 张、测试集 150 张。
- 图像尺寸：2448 x 2048。
- 类别目标数：open_solder 1080、short 1119、skewing 873、solder_bridge 600、tombstoning 880。
- 自动光照检查阈值：平均灰度低于 60 或高于 200。当前数据平均灰度约为 98-123，因此默认不修改原始图像，避免破坏已经过清洗的数据。
- 增广仅在训练阶段在线执行，验证集和测试集保持原始分布。

## GitHub 发布内容

仓库地址：https://github.com/MimiQuark/yolov8-pcba-defect-detection

提交到 Git 的内容包括源代码、配置、测试、GitHub Actions、最优 PyTorch 权重、ONNX 模型、训练与测试指标、网页截图、Word 报告和 PDF 报告。原始数据集约 700 MB、训练运行目录和预测输出不纳入 Git 历史，可通过 README 中的数据准备流程重新生成。

## 环境要求

- Windows 10/11。
- Python 3.11-3.14。
- 默认 CPU 训练，不需要 CUDA。
- 建议至少 8 GB 内存；完整 960 输入训练建议 16 GB 内存。

## 项目主页与下载

- GitHub Pages：https://mimiquark.github.io/yolov8-pcba-defect-detection/
- 源码仓库：https://github.com/MimiQuark/yolov8-pcba-defect-detection
- Release 与模型下载：https://github.com/MimiQuark/yolov8-pcba-defect-detection/releases

GitHub Pages 用于展示项目文档、指标、图表和下载入口。Gradio 交互服务依赖 Python 运行环境，不能直接托管在 GitHub Pages 上。

## 快速开始

在 PowerShell 中执行：

```powershell
cd yolov8_pcba_defect
.\scripts\setup.ps1
.\scripts\prepare.ps1
.\scripts\train_smoke.ps1
.\scripts\train_full.ps1
.\scripts\evaluate.ps1
.\scripts\launch_web.ps1
```

也可以直接使用虚拟环境中的 `pcba` 命令：

```powershell
.\.venv\Scripts\pcba.exe prepare --archive "data\source\data_y.zip" --lighting auto --overwrite
.\.venv\Scripts\pcba.exe analyze
.\.venv\Scripts\pcba.exe train --config configs\train_cpu.yaml
.\.venv\Scripts\pcba.exe eval --split test --imgsz 960
.\.venv\Scripts\pcba.exe predict --source data\pcba_yolo\images\test\351.jpg --imgsz 960
.\.venv\Scripts\pcba.exe app
```

## CLI 命令

### 数据准备

```powershell
pcba prepare --archive <data_y.zip> --output data\pcba_yolo --lighting auto
```

将 XML 的 `xmin/ymin/xmax/ymax` 转换为 YOLO 的归一化 `x_center/y_center/width/height`。如果发现图片与标注不匹配、未知类别、越界框或缺失标签，程序会立即失败并报告具体文件。

### 数据分析

```powershell
pcba analyze --dataset data\pcba_yolo --output data\analysis
```

生成：

- `summary.json`：划分、类别、目标尺寸和亮度统计。
- `class_distribution.csv`：各类别数量。
- `object_sizes.csv`：每个目标框的像素尺寸和面积占比。
- `brightness_hist.png`：亮度直方图。
- `class_distribution.png`：各类别数量图。
- `samples_annotated.jpg`：标注样例。

### 训练

```powershell
pcba train --config configs\smoke_cpu.yaml
pcba train --config configs\train_cpu.yaml
```

平衡 CPU 配置：

- 模型：`models/yolov8n.pt`（项目内已放置预训练权重）
- 输入：960
- batch：2
- epochs：40
- patience：10
- workers：2
- device：cpu
- seed：42
- 在线增广：HSV、水平翻转、轻微旋转/平移/缩放、Mosaic。
- 第 30 轮之后关闭 Mosaic，帮助模型在最后阶段稳定收敛。

如需临时缩短训练，可覆盖配置：

```powershell
pcba train --config configs\train_cpu.yaml --epochs 10 --imgsz 640 --batch 4
```

### 测试集评估

```powershell
pcba eval --split test --imgsz 960 --batch 2 --device cpu
```

输出总体和逐类别 Precision、Recall、mAP50、mAP50-95、混淆矩阵和速度统计。模型冻结后测试集只用于最终评估，不参与训练调参。

### 推理

```powershell
pcba predict --source path\to\image.jpg --conf 0.25 --iou 0.45 --imgsz 960
pcba predict --source path\to\images --imgsz 960
pcba predict --source path\to\video.mp4 --imgsz 640
pcba predict --source 0 --show
```

输出结果保存在 `outputs/predict/<时间戳>/`，包含标注文件、`summary.json` 和 `detections.csv`。

### ONNX 导出

```powershell
pcba export --output models\pcba_yolov8n.onnx --imgsz 960 --opset 17
```

导出后会使用 `onnx.checker` 和 ONNX Runtime CPU 加载验证模型。

### Web 演示

```powershell
pcba app --host 127.0.0.1 --port 7860
```

浏览器访问 `http://127.0.0.1:7860`。页面支持图片上传、摄像头、视频、置信度/IoU/输入尺寸调节、检测明细和 CSV 下载。


## 本次实验结果

- 正式训练在 CPU 上运行约 2.96 小时，第 36 轮触发早停，最佳权重来自第 26 轮。
- 测试集总体 Precision 94.18%，Recall 77.42%，mAP50 87.94%，mAP50-95 53.13%。
- 逐类别 mAP50：open_solder 82.07%，short 99.50%，skewing 80.54%，solder_bridge 99.03%，tombstoning 78.57%。
- `open_solder` 的召回率相对较低，是后续优化时最需要重点补充样本和调整阈值的类别。
- 验证集与测试集来自不同编号区间，验证指标明显低于测试指标，提示当前固定编号划分存在批次或采集条件差异；正式项目应采用分层或交叉验证划分。
- 模型、ONNX、测试指标、网页截图和报告均保存在 `models/`、`runs/eval/`、`reports/` 目录。

## 项目结构

```text
yolov8_pcba_defect/
├─ configs/                 训练配置
├─ docs/                    GitHub Pages 项目主页
├─ data/
│  ├─ pcba_yolo/            YOLO 格式数据和 dataset.yaml
│  └─ analysis/             数据统计和图表
├─ artifacts/               发布用模型、报告和评估结果
├─ outputs/                 推理结果和网页演示记录
├─ runs/                    训练和评估日志
├─ scripts/                 PowerShell 快捷脚本
├─ .github/                 GitHub Actions 配置
├─ src/pcba_defect/         项目源码
└─ tests/                   自动化测试
```

## 测试

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check src tests
```

测试覆盖坐标转换、光照处理、类别映射、合格判定、CLI 命令和已准备数据完整性。

## 常见问题

- `yolov8n.pt` 下载失败：检查网络后重新执行训练命令，Ultralytics 会自动把权重缓存到用户目录。
- Windows 多进程报错：把训练配置中的 `workers` 改为 0。
- 训练内存不足：降低 `imgsz` 和 `batch`，例如 `640 / 4`。
- CPU 训练时间过长：先使用 `configs/smoke_cpu.yaml` 验证流程，再运行完整训练。
- 中文路径读取失败：项目和 YOLO 数据统一使用英文路径；原始压缩包可以继续放在原位置。
- 网页无输出：确认 `models/pcba_yolov8n_best.pt` 或训练目录中的 `best.pt` 已生成。
