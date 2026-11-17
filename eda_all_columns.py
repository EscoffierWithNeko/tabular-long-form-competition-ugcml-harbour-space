import pandas as pd
import numpy as np

train = pd.read_csv('train.csv', nrows=5000)
test = pd.read_csv('test.csv', nrows=1000)

print(f'Train head rows: {train.shape}, Test head rows: {test.shape}')
print('\n=== COLUMNS LIST ===')
for i, col in enumerate(train.columns):
    print(f'{i:2d}: {col}')

print('\n=== SAMPLE ROW ===')
for col in train.columns:
    val = str(train[col].iloc[0])
    if len(val) > 60:
        val = val[:57] + '...'
    print(f'{col:30s}: {val}')
