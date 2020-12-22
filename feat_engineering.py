import pandas as pd
from sklearn import preprocessing

features_csv = "data/features.csv"
df = pd.read_csv(features_csv, index_col=0)
print(df.shape)

num_cols = ["word_count",
            "question_mark",
            "exclamation_mark",
            "start_digit",
            "start_question",
            "longest_word_len",
            "avg_word_len",
            "ratio_stopwords"]

df[num_cols] = preprocessing.scale(df[num_cols])

df.to_csv(path_or_buf="data/normalised.csv")
