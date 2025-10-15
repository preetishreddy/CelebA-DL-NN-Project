import json, numpy as np, pandas as pd
from pathlib import Path

def summarize_pred_jsons(pred_dir, out_csv_collage, out_csv_celeb):
    files = sorted(Path(pred_dir).glob("*_pred.json"))
    rows=[]; per_celeb={}
    for f in files:
        data=json.load(open(f))
        for g in data.get("faces",[]):
            rows.append(g)
            if g["shown"]!="unknown":
                cid=g["shown"]
                per_celeb.setdefault(cid, dict(gt=0,detected=0,correct=0,unknown_on_matched=0))
                per_celeb[cid]["detected"]+=1
                per_celeb[cid]["correct"] += int(g["shown"]==g["pred_label"])
            # NOTE: if you want GT comparison, add your GT loader here
    pd.DataFrame(rows).to_csv(out_csv_collage, index=False)

    pc=[]
    for cid,stats in per_celeb.items():
        det=stats["detected"]; corr=stats["correct"]
        pc.append(dict(celeb_id=cid, detected=det, id_acc_on_detected=(corr/det if det else 0)))
    pd.DataFrame(pc).to_csv(out_csv_celeb, index=False)
    print("Wrote:", out_csv_collage, "and", out_csv_celeb)
