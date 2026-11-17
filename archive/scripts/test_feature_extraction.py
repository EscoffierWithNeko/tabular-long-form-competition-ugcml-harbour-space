import re
import ast
import pandas as pd
import numpy as np

train = pd.read_csv('train.csv', nrows=2000)

# 1. 验证 bathrooms 解析
def parse_bath(s):
    if pd.isna(s):
        return np.nan, 0, 0
    s = str(s).lower().strip()
    shared = 1 if 'shared' in s else 0
    private = 1 if 'private' in s else 0
    m = re.search(r'([0-9]*\.?[0-9]+)', s)
    if m:
        val = float(m.group(1))
    elif 'half-bath' in s or 'half bath' in s:
        val = 0.5
    else:
        val = 1.0
    return val, shared, private

baths = train['bathrooms_text'].apply(parse_bath)
train['bath_num'] = [b[0] for b in baths]
train['is_shared_bath'] = [b[1] for b in baths]
train['is_private_bath'] = [b[2] for b in baths]
print('Bath num describe:')
print(train['bath_num'].describe())
print('Shared counts:', train['is_shared_bath'].value_counts().to_dict())

# 2. 验证 amenities 解析
key_amenities = [
    'pool', 'air conditioning', 'elevator', 'free parking', 'paid parking',
    'gym', 'hot tub', 'washer', 'dryer', 'kitchen', 'wifi', 'patio',
    'balcony', 'dishwasher', 'coffee maker', 'dedicated workspace',
    'bbq grill', 'sauna', 'waterfront', 'ev charger'
]

def extract_amenities_feats(series):
    results = {}
    # amenities length
    clean_series = series.fillna('[]').astype(str)
    results['amenities_count'] = clean_series.apply(lambda x: len(re.findall(r'"(.*?)"', x)))
    lowered = clean_series.str.lower()
    for k in key_amenities:
        col_name = 'has_' + re.sub(r'[^a-z0-9]', '_', k)
        results[col_name] = lowered.str.contains(k, regex=False).astype(int)
    return pd.DataFrame(results)

am_df = extract_amenities_feats(train['amenities'])
print('Amenities counts summary:')
print(am_df.sum())

# 3. 验证日期字段
for d_col in ['host_since', 'first_review', 'last_review']:
    train[d_col] = pd.to_datetime(train[d_col], errors='coerce')

ref_date = pd.to_datetime('2026-10-15')
print('Host since days non-null:', (ref_date - train['host_since']).dt.days.notna().sum())
print('Review span days mean:', (train['last_review'] - train['first_review']).dt.days.mean())
