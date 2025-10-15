from pathlib import Path
import yaml, json, random, numpy as np
import torch

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    # resolve paths
    p = cfg["paths"]
    cfg["paths"]["raw_dir"]       = str(Path(p["raw_dir"]).resolve())
    cfg["paths"]["landmarks_csv"] = str(Path(p["landmarks_csv"]).resolve())
    cfg["paths"]["weights_dir"]   = str(Path(p["weights_dir"]).resolve())
    cfg["paths"]["outputs"]       = str(Path(p["outputs"]).resolve())
    return cfg

def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
