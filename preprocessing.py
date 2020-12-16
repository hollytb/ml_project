import pandas as pd
import numpy as np
from nltk import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from string import punctuation

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
    return int(headline.startswith(('what', 'where', 'when', 'who', 'why', 'whom', 'whose', 'which',
                                    'how', 'will', 'would', 'should', 'could', 'do', 'did')))


def remove_punctuation(headline):
    return ''.join(w for w in headline if w not in punctuation)


def tokenize(headline):
    tokens = [word_tokenize(x) for x in headline]
    return tokens


def longest_word_len(tokens):
    """
    :param tokens: tokenized headline
    """
    token_lengths = set(len(token) for token in tokens)
    return max(token_lengths)


def avg_word_len(tokens):
    """
    :param tokens: tokenized headline
    """
    return sum(len(token) for token in tokens) / len(tokens)


def ratio_stopwords(tokens):
    """
    :param tokens: tokenized headline
    """
    stop_words = set(stopwords.words('english'))
    count = 0
    for token in tokens:
        if token in stop_words:
            count += 1

    return count / len(tokens)


def remove_stopwords(tokens):
    """
    :param tokens: tokenized headline
    """
    stop_words = set(stopwords.words('english'))
    return [token for token in tokens if token not in stop_words]


def pos_tagging(tokens):
    return pos_tag(tokens)


def lemmatize(tokens):
    lemmatizer = WordNetLemmatizer()
    new_tokens = []
    for token in tokens:
        new_tokens.append(lemmatizer.lemmatize(token[0], pos=token[1]))
    return new_tokens


# tokens for POS tagging and lemmatisation later
# df['raw_tokens'] = tokenize(df.headline)

# make lowercase
df['headline'] = df['headline'].str.lower()

# text processing with strings BEFORE removing punctuation and stopwords
df['word_count'] = df['headline'].apply(word_count)
df['question_mark'] = df['headline'].apply(question_mark)
df['exclamation_mark'] = df['headline'].apply(exclamation_mark)
df['start_digit'] = df['headline'].apply(starts_with_digit)
df['start_question'] = df['headline'].apply(starts_with_question_word)

# remove punctuation
df['headline'] = df['headline'].apply(remove_punctuation)

# tokenize
df['headline'] = tokenize(df.headline)

# text processing with tokens
df['longest_word_len'] = df['headline'].apply(longest_word_len)
df['avg_word_len'] = df['headline'].apply(avg_word_len)
df['ratio_stopwords'] = df['headline'].apply(ratio_stopwords)

# remove stopwords
df['headline'] = df['headline'].apply(remove_stopwords)

# TODO lemmatisation with POS tagging - do we need it?
# df['raw_tokens'] = df['raw_tokens'].apply(pos_tagging)
# df['raw_tokens'] = df['raw_tokens'].apply(lemmatize)


with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(df)

# final_df = df.drop(columns={'headline', 'date', 'source'})
# print(final_df)
#
# final_df.to_csv(path_or_buf="data/features.csv", index=False)
