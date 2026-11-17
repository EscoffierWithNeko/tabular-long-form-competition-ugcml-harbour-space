import pandas as pd
import numpy as np
from collections import Counter
import json
import re

df = pd.read_csv('train.csv', nrows=2000)
print("Columns:", list(df.columns))

print("\n--- Amenities sample ---")
for i, s in enumerate(df['amenities'].dropna().head(5)):
    print(f"{i+1}: {s[:150]}...")

# Count top amenities
all_amenities = []
for s in df['amenities'].dropna():
    # amenities are typically JSON array like ["Wifi", "Kitchen", ...]
    try:
        items = json.loads(s)
    except Exception:
        # fallback regex
        items = re.findall(r'"([^"]+)"', s)
    for it in items:
        clean_it = it.strip().lower()
        if clean_it:
            all_amenities.append(clean_it)

counter = Counter(all_amenities)
print(f"\nTotal unique amenities in 2000 samples: {len(counter)}")
print("Top 40 most common amenities:")
for item, cnt in counter.most_common(40):
    print(f"  {item}: {cnt}")

print("\n--- Name and Description sample ---")
for i, (nm, desc) in enumerate(zip(df['name'].dropna().head(3), df['description'].dropna().head(3))):
    print(f"Name {i+1}: {nm}")
    print(f"Desc {i+1}: {str(desc)[:120]}...")
