import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

raw_feats_csv = "data/features.csv"
df = pd.read_csv(raw_feats_csv)
print(df.head())

data = np.array(df.iloc[:, 1:])
labels = np.array(df.iloc[:, 0])
labels = labels.reshape(-1, 1)

scaler = MinMaxScaler()
scaler.fit(data)
scaled_data = scaler.transform(data)

df = pd.DataFrame(scaled_data)
df.insert(0, None, labels)

df.to_csv(path_or_buf="data/normalised.csv", index=False, header=False)
