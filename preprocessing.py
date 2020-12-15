import pandas as pd
import numpy as np

csv_file = "data/small_set.csv"
df = pd.read_csv(csv_file, comment='#', quotechar='"', skipinitialspace=True, index_col=0)
print(df.head())
headlines = np.array(df.iloc[:, 0])
headlines = headlines.reshape(-1, 1)
y = np.array(df.iloc[:, 2])
y = y.reshape(-1, 1)


def word_count(headline):
    return len(headline.split())


def question_mark(headline):
    return int('?' in headline)


def exclamation_mark(headline):
    return int('!' in headline)


df['word_count'] = df['headline'].apply(word_count)
df['question'] = df['headline'].apply(question_mark)
df['exclamtion'] = df['headline'].apply(exclamation_mark)

print(df.head())
