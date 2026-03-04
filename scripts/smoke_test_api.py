# scripts/smoke_test_api.py
import json
import requests

schema = json.load(open("artifacts/feature_schema.json"))
cols = schema["feature_columns"]

# make a fake single row with zeros (good for smoke test)
row = {"group": "dfs.FSDataset", "window_start": "2008-11-11 04:45:00", "features": {c: 0.0 for c in cols}}

r = requests.post("http://127.0.0.1:8000/score_batch", json={"model": "iforest", "rows": [row]})
print(r.status_code)
print(r.json())

print("metrics:", requests.get("http://127.0.0.1:8000/metrics").json())