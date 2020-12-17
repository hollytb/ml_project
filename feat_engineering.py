import pandas as pd

raw_feats_csv = "data/features.csv"
df = pd.read_csv(raw_feats_csv)
print(df.head())

# TODO normalisation of all feature columns - what type will we use?