from PIL import Image
from pathlib import Path
import random

def dump_face_chips(processed_train, chips_dir, per_class=10):
    chips_dir.mkdir(parents=True, exist_ok=True)
    for cid_dir in sorted(processed_train.glob("*")):
        ims = sorted(cid_dir.glob("*.jpg"))
        random.shuffle(ims)
        for p in ims[:per_class]:
            out = chips_dir/f"{cid_dir.name}__{p.name}"
            if not out.exists():
                Image.open(p).convert("RGB").resize((128,128), Image.BILINEAR).save(out, quality=95)

def baseline_collage(chips_dir, out_path, grid=(4,4), cell=128, pad=8, bg=32):
    W = grid[1]*cell + (grid[1]+1)*pad
    H = grid[0]*cell + (grid[0]+1)*pad
    im = Image.new("RGB",(W,H),(bg,bg,bg))
    chips = list(chips_dir.glob("*.jpg")); random.shuffle(chips)
    k=0
    for r in range(grid[0]):
        for c in range(grid[1]):
            if k>=len(chips): break
            tile = Image.open(chips[k]).resize((cell,cell), Image.BILINEAR)
            x = pad + c*(cell+pad); y = pad + r*(cell+pad)
            im.paste(tile,(x,y)); k+=1
    im.save(out_path, quality=95)

def random_collage(chips_dir, out_path, canvas=(1200,800), n_tiles=(15,30)):
    W,H = canvas
    im = Image.new("RGB",(W,H),(32,32,32))
    chips = list(chips_dir.glob("*.jpg")); random.shuffle(chips)
    T = random.randint(*n_tiles)
    for p in chips[:T]:
        t = Image.open(p).convert("RGB")
        s = random.randint(110,200)
        t = t.resize((s,s), Image.BILINEAR)
        t = t.rotate(random.uniform(-17,17), expand=True, resample=Image.BILINEAR)
        x = random.randint(0, max(0, W - t.width))
        y = random.randint(0, max(0, H - t.height))
        im.paste(t,(x,y))
    im.save(out_path, quality=95)
