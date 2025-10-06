#  YOLOv8 Face Identification Pipeline

This project implements a synthetic data generation and training pipeline to identify celebrity faces using YOLOv8.
It uses 40 celebrity image folders to create group photos with annotated bounding boxes, trains a custom YOLOv8 model, and evaluates its performance.

---

##  Project Overview

The pipeline automates the entire flow from data preparation to model training:

1. **Data Loading** – Loads multiple celebrity folders from Google Drive (each folder = one celebrity, containing ~30 images).
2. **Synthetic Data Generation** – Creates group photos by concatenating 5–40 faces randomly on a blank canvas (1080×720).
3. **Annotation Creation** – Generates YOLO-format `.txt` label files for each group image with bounding boxes and class IDs.
4. **Augmentation** – Applies random transformations (flip, brightness, rotation) to increase dataset diversity.
5. **YOLOv8 Training** – Fine-tunes the pretrained `yolov8n.pt` model on the generated dataset.
6. **Model Export** – Saves the best-trained model as `best.pt` for later inference.

---

##  Folder Structure

```
project_root/
│
├── content/
│   ├── group_celeb.yaml                # YOLO dataset config
│   ├── yolov8n.pt                      # Pretrained YOLOv8 weights
│   └── runs/
│       └── detect/train/weights/       # Trained model checkpoints
│           ├── best.pt
│           └── last.pt
│
├── datasets/
│   └── content/datasets/group_celeb/
│       ├── images/
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       └── labels/
│           ├── train/
│           ├── val/
│           └── test/
│
└── face_detection_pipeline.py          # Complete pipeline script
```

---

##  Setup Instructions

### 1️⃣ Install Dependencies

Run this inside Colab or a Python environment:

```bash
pip install ultralytics opencv-python pillow torchvision tqdm pyyaml
```

### 2️⃣ Mount Google Drive (Colab)

```python
from google.colab import drive
drive.mount('/content/drive')
```

### 3️⃣ Load Data

Ensure your dataset folder (`Celebrity Image Subsets`) is in your Drive under:

```
/content/drive/MyDrive/Celebrity Image Subsets
```

Each subfolder should follow:

```
images_<id>/
    1.jpg
    2.jpg
    ...
```

### 4️⃣ Run the Synthetic Dataset Generator

Creates 10,000 group images and YOLO annotations:

```python
# run the generator cell
```

### 5️⃣ Train YOLOv8

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.train(
    data="/content/group_celeb.yaml",
    epochs=25,
    imgsz=640,
    batch=16,
    workers=2,
    device=0  # GPU
)
```

---

##  Model Details

* **Base Model**: YOLOv8n (pretrained on COCO dataset)
* **Task**: Object detection (face + identity classification)
* **Classes**: 43 celebrities (auto-mapped numeric IDs)
* **Augmentations**: Horizontal flip, rotation, color jitter
* **Canvas Size**: 1280×720
* **Face Size**: 128×128
* **Overlap tolerance**: ≤ 5%

---

##  Training Summary

| Metric           | Value                   |
| ---------------- | ----------------------- |
| Epochs           | 25                      |
| Dataset Size     | 10,000 synthetic images |
| GPU Used         | Colab A100 / T4 GPU     |
| mAP50            | 0.995                   |
| Precision/Recall | 0.999 / 0.999           |

---

##  Model Output

After training:

```
runs/detect/train/weights/best.pt
```

You can download it:

```python
from google.colab import files
files.download("runs/detect/train/weights/best.pt")
```

---

##  Inference Example

```python
from ultralytics import YOLO
model = YOLO("best.pt")

results = model.predict(source="group_photo.jpg", show=True)
```

YOLO will output bounding boxes labeled with celebrity IDs.

---



