import re
import json
import numpy as np
import pandas as pd

def parse_bathrooms(text):
    if pd.isna(text):
        return np.nan, 0, 0
    text_str = str(text).lower().strip()
    is_shared = 1 if 'shared' in text_str else 0
    is_private = 1 if 'private' in text_str else 0
    
    # 查找数值
    match = re.search(r'([0-9]*\.?[0-9]+)', text_str)
    if match:
        val = float(match.group(1))
    elif 'half-bath' in text_str or 'half bath' in text_str:
        val = 0.5
    else:
        val = 1.0
    return val, is_shared, is_private

def test():
    samples = [
        '1 bath', '2 baths', '1 shared bath', '1 private bath',
        '1.5 baths', '2.5 baths', 'Shared half-bath', '0 baths',
        'Half-bath', None, np.nan, 'Private half-bath'
    ]
    for s in samples:
        val, shared, priv = parse_bathrooms(s)
        print(f'{str(s):<20} -> val={val}, shared={shared}, priv={priv}')

if __name__ == '__main__':
    test()
