import argparse, json
from pathlib import Path
import torch
from PIL import Image, ImageDraw, ImageFont

from src.celebnet.config import load_config, set_seed
from src.celebnet.landmarks import load_landmarks
from src.celebnet.split_align import split_raw, write_aligned
from src.celebnet.datasets import build_loaders
from src.celebnet.model import build_backbone, build_head
from src.celebnet.train import train
from src.celebnet.calibrate import calibrate_temperature
from src.celebnet.collages import dump_face_chips, baseline_collage, random_collage
from src.celebnet.detect_identify import (
    load_labels, idx2label_map, load_calibration,
    setup_detector, detect_boxes, pad_square, classify_chip
)
from src.celebnet.eval_collages import summarize_pred_jsons


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["split", "align", "train", "calib", "collages", "infer_dir", "eval"])
    ap.add_argument("--cfg", default="config.yaml")
    ap.add_argument("--inp", help="input image dir for infer_dir", default=None)
    args = ap.parse_args()

    cfg = load_config(args.cfg)
    set_seed(cfg["seed"])

    P = cfg["paths"]
    OUT = Path(P["outputs"])
    PROCESSED = OUT / "processed"
    MODELS = OUT / "models"
    PLOTS = OUT / "plots"
    COLLAGES = OUT / "collages"

    # ------------------ SPLIT ------------------
    if args.cmd == "split":
        split_raw(Path(P["raw_dir"]), cfg["splits"], OUT)
        print("✅ Split CSVs written to", OUT)

    # ------------------ ALIGN ------------------
    elif args.cmd == "align":
        lm = load_landmarks(P["landmarks_csv"])
        for s in ["train", "val", "test"]:
            stats = write_aligned(OUT / f"{s}_manifest.csv", lm, s, PROCESSED, cfg["input_size"])
            print(s, stats)

    # ------------------ TRAIN ------------------
    elif args.cmd == "train":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        loaders, label2idx = build_loaders(PROCESSED, batch=cfg["train"]["batch_size"])
        backbone = build_backbone(device)
        head = build_head(num_classes=len(label2idx), device=device)
        train(backbone, head, loaders, cfg, MODELS, PLOTS, device)

    # ------------------ CALIBRATION ------------------
    elif args.cmd == "calib":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        loaders, label2idx = build_loaders(PROCESSED, batch=cfg["train"]["batch_size"])
        backbone = build_backbone(device)
        head = build_head(num_classes=len(label2idx), device=device)
        ckpt = torch.load(MODELS / "faceid_linear_head.pt", map_location=device)
        head.load_state_dict(ckpt["classifier_state"])
        head.eval()
        calibrate_temperature(backbone, head, loaders["val"], MODELS, device)

    # ------------------ COLLAGE CREATION ------------------
    elif args.cmd == "collages":
        dump_face_chips(PROCESSED / "train", COLLAGES / "face_chips", per_class=10)
        baseline_collage(COLLAGES / "face_chips", COLLAGES / "baseline" / "sample_baseline.jpg")
        random_collage(COLLAGES / "face_chips", COLLAGES / "random" / "sample_random.jpg")
        print("✅ Sample collages written.")

    # ------------------ INFERENCE ON DIRECTORY ------------------
    elif args.cmd == "infer_dir":
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # --- Load trained model + labels (keys cleaned to numeric IDs) ---
        label2idx = load_labels(MODELS / "labels.json")
        idx2label = idx2label_map(label2idx)

        backbone = build_backbone(device)
        head = build_head(num_classes=len(label2idx), device=device)
        ckpt = torch.load(MODELS / "faceid_linear_head.pt", map_location=device)
        head.load_state_dict(ckpt["classifier_state"])
        head.eval()

        # --- Calibration + detector setup ---
        T = load_calibration(MODELS / "calibration.json")
        kind, detector = setup_detector(P["weights_dir"])
        prob_thr  = cfg["infer"]["prob_thresh"]
        margin_thr= cfg["infer"]["margin_thresh"]
        pad_frac  = cfg["infer"]["pad_frac"]

        # --- Input and output directories ---
        if not args.inp:
            raise SystemExit("❌ Please specify input folder: python cli.py infer_dir --inp path/to/folder")
        in_dir = Path(args.inp)
        if not in_dir.exists() or not in_dir.is_dir():
            raise SystemExit(f"❌ Input directory not found: {in_dir}")

        # Save to outputs/collages/<input_folder_name>_preds/
        pred_dir_name = f"{in_dir.name}_preds"
        out_dir = OUT / "collages" / pred_dir_name
        out_dir.mkdir(parents=True, exist_ok=True)

        # Gather images
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        img_paths = sorted([p for p in in_dir.iterdir() if p.suffix.lower() in exts])
        if not img_paths:
            raise SystemExit(f"❌ No images found in {in_dir} (supported: {sorted(exts)})")

        # Inference loop
        for p in img_paths:
            im = Image.open(p).convert("RGB")
            boxes = detect_boxes(detector, kind, im)
            draw = ImageDraw.Draw(im)
            rows = []

            for (x1, y1, x2, y2) in boxes:
                chip = pad_square(im, x1, y1, x2, y2, pad=pad_frac, out=cfg["input_size"])
                idx, p1, margin = classify_chip(backbone, head, chip, T, device)
                lab = idx2label[idx]  # numeric celeb ID
                shown = lab if (p1 >= prob_thr and margin >= margin_thr) else "unknown"
                color = (40, 220, 100) if shown != "unknown" else (240, 110, 80)
                draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                draw.text((x1 + 3, max(0, y1 - 16)), f"{shown} p={p1:.2f} Δ={margin:.2f}", fill=color)

                rows.append({
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "pred_label": lab,
                    "shown": shown,
                    "conf": float(p1),
                    "margin": float(margin),
                })

            out_img  = out_dir / f"{p.stem}_pred.jpg"
            out_json = out_dir / f"{p.stem}_pred.json"
            im.save(out_img, quality=95)
            json.dump({"collage": str(p), "faces": rows}, open(out_json, "w"), indent=2)
            print(f"✅ {p.name} → {out_img.name}")

        print(f"\n✅ All predictions saved to: {out_dir}")

    # ------------------ EVALUATION ------------------
    elif args.cmd == "eval":
        rp = OUT / "collages" / "random_preds"
        summarize_pred_jsons(rp, rp / "random_eval_per_collage.csv", rp / "random_eval_per_celeb.csv")


if __name__ == "__main__":
    main()

