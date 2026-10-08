# Model Card YOLOv8n PCBA Defect Detector

## Model Details

- Architecture: Ultralytics YOLOv8n detection model.
- Task: Five-class PCBA solder defect detection.
- Input: RGB image, trained at 960 x 960.
- Output: Bounding boxes, class IDs, and confidence scores.
- Classes: `open_solder`, `short`, `skewing`, `solder_bridge`, `tombstoning`.

## Intended Use

The model is intended for course projects, offline inspection demonstrations, and further research. It is not approved for autonomous production-line safety decisions.

## Training Data

- 600 images and 600 Pascal VOC XML files.
- 4552 annotated objects.
- Original split: 350 train, 100 validation, 150 test.
- Image size: 2448 x 2048.
- Automatic brightness analysis found no images outside the 60-200 mean-grayscale range, so no offline brightness correction was applied.

## Evaluation

Final evaluation uses the held-out 150-image test split at 960 input resolution.

- Precision: 0.9418
- Recall: 0.7742
- mAP50: 0.8794
- mAP50-95: 0.5313

`open_solder` has the lowest recall at 54.59% and should be prioritized in future data collection and threshold tuning.

## Limitations

- The validation and test splits use different image-number ranges, so their metrics differ substantially. This may indicate acquisition-batch differences.
- Small and visually similar defects can still be missed.
- Performance on new lighting, camera, board layout, or production-line conditions is not guaranteed.
- CPU inference at 960 is suitable for demonstration but may be too slow for strict real-time sorting.

## Reproducibility

Use the CLI documented in the repository README. The best PyTorch and ONNX weights, training configuration, metrics, and report are included under `artifacts/`.
