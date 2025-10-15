import pandas as pd

def load_landmarks(csv_path):
    df_raw = pd.read_csv(csv_path)
    cols = {c.lower(): c for c in df_raw.columns}
    mapping = dict(
        image_name = cols.get("image_id") or cols.get("image_name") or list(df_raw.columns)[0],
        le_x=cols.get("lefteye_x"), le_y=cols.get("lefteye_y"),
        re_x=cols.get("righteye_x"), re_y=cols.get("righteye_y"),
        n_x=cols.get("nose_x"),     n_y=cols.get("nose_y"),
        ml_x=cols.get("leftmouth_x"), ml_y=cols.get("leftmouth_y"),
        mr_x=cols.get("rightmouth_x"), mr_y=cols.get("rightmouth_y"),
    )
    missing = [k for k,v in mapping.items() if v is None]
    if missing:
        raise ValueError(f"Missing landmark columns in CSV: {missing}")
    df = pd.DataFrame({k: df_raw[v] for k,v in mapping.items()})
    df["key"] = df["image_name"].astype(str)
    return df.set_index("key").to_dict(orient="index")
