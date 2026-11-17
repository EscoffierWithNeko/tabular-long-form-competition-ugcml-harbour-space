import pandas as pd

train = pd.read_csv('train.csv', nrows=2)
print("Columns in train.csv:")
for i, col in enumerate(train.columns):
    print(f"{i}: {col}")
