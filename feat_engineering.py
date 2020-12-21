import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

raw_feats_csv = "data/test.csv"
df = pd.read_csv(raw_feats_csv)
print(df.head())

tokens = df.iloc[:, 1]
data = df.iloc[:, 2:]
print(data)
# labels = np.array(df.iloc[:, 0])
# labels = labels.reshape(-1, 1)

scaler = MinMaxScaler()
scaler.fit(data)
data = scaler.transform(data)

df = pd.DataFrame(data)
df.insert(0, 'text', tokens)
df.insert(0, 'class', 0)

df.to_csv(path_or_buf="data/normalised.csv")
