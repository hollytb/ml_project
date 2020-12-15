import pandas as pd
import numpy as np
from nltk import word_tokenize

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


def starts_with_digit(headline):
    return int(headline[0].isdigit())


def starts_with_question_word(headline):
    return int(headline.startswith(('what', 'where', 'when', 'who', 'why', 'whom', 'whose', 'which', 'how',)))
    # ??? also include ,'will','would','should','could','do','did' ???


def longest_word_len(headline):
    tokens = word_tokenize(headline)
    token_lengths = set(len(token) for token in tokens)
    return max(token_lengths)


# processing before making lowercase? eg count num of words with cap letter start

df['headline'] = df['headline'].str.lower()

df['word_count'] = df['headline'].apply(word_count)
df['question'] = df['headline'].apply(question_mark)
df['exclamation'] = df['headline'].apply(exclamation_mark)
df['start_digit'] = df['headline'].apply(starts_with_digit)
df['start_question'] = df['headline'].apply(starts_with_question_word)
df['longest_word_len'] = df['headline'].apply(longest_word_len)

print(df.head())
