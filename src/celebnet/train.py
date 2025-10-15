import torch, torch.nn as nn, torch.optim as optim
import numpy as np, json, matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from pathlib import Path
from .model import build_backbone, build_head

def run_epoch(backbone, head, loader, device, train=False):
    if train: head.train()
    else:     head.eval()
    crit = nn.CrossEntropyLoss()
    opt  = getattr(run_epoch, "_opt", None)

    if train and opt is None:
        raise RuntimeError("Optimizer not set on run_epoch")

    losses=[]; preds=[]; gts=[]
    for x,y in loader:
        x=x.to(device); y=y.to(device)
        with torch.set_grad_enabled(train):
            emb = backbone(x)
            logits = head(emb)
            loss = crit(logits, y)
        if train:
            opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
        preds.extend(torch.argmax(logits,1).cpu().tolist())
        gts.extend(y.cpu().tolist())
    return float(np.mean(losses)), accuracy_score(gts,preds)

def train(backbone, head, loaders, cfg, out_models: Path, out_plots: Path, device):
    lr=cfg["train"]["lr"]; wd=cfg["train"]["weight_decay"]
    epochs=cfg["train"]["epochs"]; patience=cfg["train"]["patience"]

    opt = optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    run_epoch._opt = opt  # stash

    best=0; wait=0
    history=dict(tr_loss=[],tr_acc=[],va_loss=[],va_acc=[])

    for ep in range(1, epochs+1):
        tl,ta = run_epoch(backbone, head, loaders["train"], device, True)
        vl,va = run_epoch(backbone, head, loaders["val"], device, False)
        history["tr_loss"].append(tl); history["tr_acc"].append(ta)
        history["va_loss"].append(vl); history["va_acc"].append(va)
        print(f"Epoch {ep:02d} | train {ta:.3f} | val {va:.3f}")

        if va>best:
            best=va; wait=0
            out_models.mkdir(parents=True, exist_ok=True)
            torch.save({"classifier_state": head.state_dict()}, out_models/"faceid_linear_head.pt")
        else:
            wait+=1
            if wait>=patience:
                print("Early stopping."); break

    # plots
    out_plots.mkdir(parents=True, exist_ok=True)
    fig,ax = plt.subplots(1,2, figsize=(11,4))
    ax[0].plot(history["tr_loss"], label="train"); ax[0].plot(history["va_loss"], label="val")
    ax[0].set_title("Loss"); ax[0].legend(); ax[0].grid(True, alpha=0.3)
    ax[1].plot(history["tr_acc"], label="train"); ax[1].plot(history["va_acc"], label="val")
    ax[1].set_title("Accuracy"); ax[1].legend(); ax[1].grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(out_plots/"training_curves.png", dpi=140)
    plt.close(fig)

    # test best
    ckpt = torch.load(out_models/"faceid_linear_head.pt", map_location=device)
    head.load_state_dict(ckpt["classifier_state"]); head.eval()
    _, test_acc = run_epoch(backbone, head, loaders["test"], device, False)
    print("Test accuracy:", round(test_acc,3))
    with open(out_models/"metrics.json","w") as f:
        json.dump({"best_val_acc": float(best), "test_acc": float(test_acc)}, f, indent=2)
