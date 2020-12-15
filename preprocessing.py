import pandas as pd
import numpy as np

csv_file = "data/small_set.csv"
df = pd.read_csv(csv_file, comment='#', header=None, quotechar='"', skipinitialspace=True)
print(df.head())
X = np.array(df.iloc[:, 0])
X = X.reshape(-1, 1)
y = np.array(df.iloc[:, 2])
y = y.reshape(-1, 1)

