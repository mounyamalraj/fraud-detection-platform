import json
import sys
import time

import pandas as pd
import requests

URL = "http://127.0.0.1:8000/score"
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 200
DELAY = float(sys.argv[2]) if len(sys.argv) > 2 else 0.1

df = pd.read_csv("data/sample_transactions.csv",low_memory=False).head(LIMIT)
meta_cols = ["isFraud", "TransactionID", "TransactionDT"]

tp = fp = fn = tn = 0
print(f"Replaying {len(df)} transactions to {URL}\n")

for _, row in df.iterrows():
    feats = json.loads(row.drop(labels=meta_cols).dropna().to_json())
    resp = requests.post(URL, json={"features": feats}, timeout=10).json()
    flagged = resp["decision"] == "review"
    actual = int(row["isFraud"])

    if flagged and actual:
        tp += 1
    elif flagged:
        fp += 1
    elif actual:
        fn += 1
    else:
        tn += 1

    tag = "ALERT" if flagged else "ok   "
    truth = "FRAUD" if actual else "legit"
    print(f"{int(row['TransactionID'])}  ${row['TransactionAmt']:>8.2f}  "
          f"score={resp['fraud_score']:.3f}  {tag}  (actual: {truth})")
    time.sleep(DELAY)

print("\n--- Summary ---")
print(f"Sent: {len(df)}   Flagged: {tp + fp}")
print(f"Fraud caught: {tp}   Missed fraud: {fn}   False alarms: {fp}")
if tp + fp:
    print(f"Precision: {tp / (tp + fp):.2f}")
if tp + fn:
    print(f"Recall:    {tp / (tp + fn):.2f}")