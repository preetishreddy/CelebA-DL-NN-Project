import torch, json, numpy as np

@torch.no_grad()
def collect_logits(backbone, head, loader, device):
    L=[]
    for x,y in loader:
        emb = backbone(x.to(device))
        L.append(head(emb).cpu())
    return torch.cat(L,0)

def calibrate_temperature(backbone, head, val_loader, out_models, device):
    logits = collect_logits(backbone, head, val_loader, device)
    def score(T):
        p = torch.softmax(logits/T, dim=1).numpy()
        return float(np.mean(p.max(1)))
    Ts = np.linspace(0.05, 1.0, 40)
    best_T = float(Ts[int(np.argmax([score(t) for t in Ts]))])
    with open(out_models/"calibration.json","w") as f:
        json.dump({"temperature": best_T}, f, indent=2)
    print("Saved calibration.json with T* =", best_T)
