# src/celebnet/detect_identify.py
from pathlib import Path
import json, re, numpy as np, torch
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO
from facenet_pytorch import MTCNN


# ---------- labels / calibration ----------

def _clean_key(s: str) -> str:
    """
    Ensure label keys are numeric-only strings.
    Accepts already-clean keys or folder-ish names (e.g., 'images__4561_').
    """
    nums = re.findall(r"\d+", s)
    return nums[-1] if nums else s


def load_labels(labels_json: Path):
    """
    Loads labels.json and guarantees keys are cleaned.
    Returns dict {<numeric_id>: index}.
    """
    labels_json = Path(labels_json)
    with open(labels_json) as f:
        raw = json.load(f)
    return {_clean_key(k): int(v) for k, v in raw.items()}


def idx2label_map(label2idx: dict):
    """Reverse mapping: index -> <numeric_id> (str)."""
    return {v: k for k, v in label2idx.items()}


def load_calibration(calib_json: Path) -> float:
    """Return temperature T*, default 1.0 if missing."""
    calib_json = Path(calib_json)
    try:
        with open(calib_json) as f:
            return float(json.load(f)["temperature"])
    except Exception:
        return 1.0


# ---------- detection ----------

def setup_detector(weights_dir: Path):
    """
    If weights/yolov8n-face.pt exists, use YOLO; otherwise fallback to MTCNN.
    Returns ('yolo'|'mtcnn', detector_object)
    """
    weights_dir = Path(weights_dir)
    w = weights_dir / "yolov8n-face.pt"
    if w.exists():
        return "yolo", YOLO(str(w))
    return "mtcnn", MTCNN(keep_all=True, device="cuda" if torch.cuda.is_available() else "cpu")


def detect_boxes(detector, kind: str, pil_img: Image.Image):
    if kind == "yolo":
        r = detector(pil_img, verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            return []
        return r.boxes.xyxy.cpu().numpy().tolist()
    else:
        boxes, _ = detector.detect(pil_img)
        if boxes is None:
            return []
        return boxes.tolist()


# ---------- cropping / transform ----------

def pad_square(im: Image.Image, x1, y1, x2, y2, pad=0.3, out=160):
    W, H = im.size
    w = x2 - x1; h = y2 - y1
    cx = (x1 + x2) / 2; cy = (y1 + y2) / 2
    side = max(w * (1 + pad), h * (1 + pad))
    nx1 = max(0, int(cx - side / 2)); ny1 = max(0, int(cy - side / 2))
    nx2 = min(W, int(cx + side / 2)); ny2 = min(H, int(cy + side / 2))
    crop = im.crop((nx1, ny1, nx2, ny2))
    cw, ch = crop.size; s = min(cw, ch)
    L = (cw - s) // 2; T = (ch - s) // 2
    return crop.crop((L, T, L + s, T + s)).resize((out, out), Image.BILINEAR)


def make_eval_tf():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])


# ---------- classifier ----------

@torch.no_grad()
def classify_chip(backbone, head, chip: Image.Image, T: float, device: str):
    x = make_eval_tf()(chip).unsqueeze(0).to(device)
    emb = backbone(x)
    logits = head(emb) / T
    probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    idx = int(np.argmax(probs)); p1 = float(probs[idx])
    p2 = float(np.partition(probs.flatten(), -2)[-2]) if probs.size > 1 else 0.0
    return idx, p1, (p1 - p2)
