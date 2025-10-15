from pathlib import Path
from collections import defaultdict
from PIL import Image
import pandas as pd
import numpy as np, cv2, random
from tqdm import tqdm

def list_images(d: Path):
    return [p for p in sorted(d.glob("*")) if p.suffix.lower() in {".jpg",".jpeg",".png",".bmp"}]

def split_raw(raw_dir: Path, ratios: dict, out_dir: Path):
    raw_dir = Path(raw_dir); out_dir = Path(out_dir)
    alld = [d for d in sorted(raw_dir.iterdir()) if d.is_dir()]
    by_id = defaultdict(list)
    for cd in alld:
        for im in list_images(cd):
            by_id[cd.name].append(str(im))

    splits = {k: [] for k in ["train","val","test","collage"]}
    for cid, lst in by_id.items():
        lst = sorted(lst); random.shuffle(lst)
        n = len(lst); a = int(ratios["train"]*n); b = int(ratios["val"]*n); c = int(ratios["test"]*n)
        parts = dict(
            train=lst[:a],
            val=lst[a:a+b],
            test=lst[a+b:a+b+c],
            collage=lst[a+b+c:]
        )
        for k, paths in parts.items():
            for p in paths:
                splits[k].append({"celeb_id": cid, "path": p})

    out_dir.mkdir(parents=True, exist_ok=True)
    for k, rows in splits.items():
        pd.DataFrame(rows).to_csv(out_dir/f"{k}_manifest.csv", index=False)
    return splits

def align_from_landmarks(img_path, lm, out_size=160):
    img = cv2.imread(str(img_path))
    if img is None: return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    le = np.array([lm['le_x'], lm['le_y']], dtype=np.float32)
    re = np.array([lm['re_x'], lm['re_y']], dtype=np.float32)
    delta = re - le
    angle = np.degrees(np.arctan2(delta[1], delta[0]))
    center = ((le + re)/2).tolist()
    M = cv2.getRotationMatrix2D(tuple(center), angle, 1.0)
    rot = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
    cx, cy = map(int, center)
    half = int(0.6 * max(np.linalg.norm(delta), 60))
    x1 = max(0, cx-half); y1 = max(0, cy-half)
    x2 = min(rot.shape[1], cx+half); y2 = min(rot.shape[0], cy+half)
    crop = rot[y1:y2, x1:x2]
    if crop.size == 0: crop = rot
    chip = cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
    return Image.fromarray(chip)

def write_aligned(manifest_csv: Path, lm_index: dict, split_name: str, out_root: Path, out_size=160):
    df = pd.read_csv(manifest_csv)
    ok = miss = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Align {split_name}"):
        p = Path(row["path"]); key = p.name
        if key not in lm_index:
            miss += 1; continue
        chip = align_from_landmarks(p, lm_index[key], out_size=out_size)
        out_dir = out_root / split_name / str(row["celeb_id"])
        out_dir.mkdir(parents=True, exist_ok=True)
        chip.save(out_dir/p.name, quality=95); ok += 1
    return dict(ok=ok, missing=miss)
