import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI
from pydantic import BaseModel

MODELS = Path(__file__).resolve().parent.parent / "models"
meta = json.loads((MODELS / "model_meta.json").read_text())
FEATURES = meta["features"]
CATS = meta["categories"]

model = xgb.XGBClassifier(enable_categorical=True)
model.load_model(str(MODELS / "xgb_v2.json"))

app = FastAPI(title="Fraud Scoring API")


class Transaction(BaseModel):
    features: dict


@app.get("/health")
def health():
    return {"status": "ok", "n_features": len(FEATURES)}


@app.post("/score")
def score(tx: Transaction):
    row = {f: tx.features.get(f, np.nan) for f in FEATURES}
    df = pd.DataFrame([row])
    for c in FEATURES:
        if c in CATS:
            df[c] = pd.Categorical(df[c], categories=CATS[c])
        else:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    p = float(model.predict_proba(df[FEATURES])[:, 1][0])
    return {"fraud_score": round(p, 4), "decision": "review" if p >= 0.5 else "approve"}