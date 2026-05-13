import bz2
import json

with bz2.open("full_raw_data/train_period_data.jsonlist.bz2", "rt") as f:
    data = [json.loads(line) for line in f]

print(f"Loaded {len(data)} records")
print(data[0].keys())
