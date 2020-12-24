import numpy as np

import pandas as pd

years=[2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020]

for YEAR in years:
	cy = '/Users/rossmccrann/MACHINE LEARNING/Group_Project/ml_finale/ml_project/NON_CLICKBAIT_DATA/IRISH_TIMES_DATA/irishtimes_data_'+ str(YEAR) + '.csv'
	df1 = pd.read_csv(cy, dtype='string', header=0)
	#print(df1.head())

	#df1.drop(columns=['source', 'value', 'date'], inplace=True)
	df1.drop(columns=['date'], inplace=True)

	print(df1.head())
	df1.dropna(subset=['text'], inplace=True)
	df1.to_csv(path_or_buf=cy, index=0)

import pandas as pd
from nltk import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from string import punctuation

pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 20)

my_punctuation = '€£' + punctuation


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
    return ''.join(w for w in headline if w not in my_punctuation)


def remove_punct_tokens(tokens):
    return [token for token in tokens if token.isalnum()]


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
    return round(sum(len(token) for token in tokens) / len(tokens), 4)


def ratio_stopwords(tokens):
    """
    :param tokens: tokenized headline
    """
    stop_words = set(stopwords.words('english'))
    count = 0
    for token in tokens:
        if token in stop_words:
            count += 1

    return round(count / len(tokens), 4)


def remove_stopwords(tokens):
    """
    :param tokens: tokenized headline
    """
    stop_words = set(stopwords.words('english'))
    return [token for token in tokens if token not in stop_words]


def pos_tagging(tokens):
    return pos_tag(tokens)


def lemmatise(tokens_wordnet):
    lemmatiser = WordNetLemmatizer()
    new_tokens = []
    for token, tag in tokens_wordnet:
        if tag is None:
            new_tokens.append(lemmatiser.lemmatize(token))
        else:
            new_tokens.append(lemmatiser.lemmatize(token, tag))
    return new_tokens


def pos_tag_convert(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


def change_pos_tag(list_tuples):
    result = []
    for token, tag in list_tuples:
        result.append(tuple([token, pos_tag_convert(tag)]))
    return result



yearly = [
'/Users/rossmccrann/MACHINE LEARNING/Group_Project/ml_finale/ml_project/NON_CLICKBAIT_DATA/IRISH_TIMES_DATA/irishtimes_data_2010.csv',
'/Users/rossmccrann/MACHINE LEARNING/Group_Project/ml_finale/ml_project/NON_CLICKBAIT_DATA/IRISH_TIMES_DATA/irishtimes_data_2011.csv',
'/Users/rossmccrann/MACHINE LEARNING/Group_Project/ml_finale/ml_project/NON_CLICKBAIT_DATA/IRISH_TIMES_DATA/irishtimes_data_2012.csv',
'/Users/rossmccrann/MACHINE LEARNING/Group_Project/ml_finale/ml_project/NON_CLICKBAIT_DATA/IRISH_TIMES_DATA/irishtimes_data_2013.csv',
'/Users/rossmccrann/MACHINE LEARNING/Group_Project/ml_finale/ml_project/NON_CLICKBAIT_DATA/IRISH_TIMES_DATA/irishtimes_data_2014.csv',
'/Users/rossmccrann/MACHINE LEARNING/Group_Project/ml_finale/ml_project/NON_CLICKBAIT_DATA/IRISH_TIMES_DATA/irishtimes_data_2015.csv',
'/Users/rossmccrann/MACHINE LEARNING/Group_Project/ml_finale/ml_project/NON_CLICKBAIT_DATA/IRISH_TIMES_DATA/irishtimes_data_2016.csv',
'/Users/rossmccrann/MACHINE LEARNING/Group_Project/ml_finale/ml_project/NON_CLICKBAIT_DATA/IRISH_TIMES_DATA/irishtimes_data_2017.csv',
'/Users/rossmccrann/MACHINE LEARNING/Group_Project/ml_finale/ml_project/NON_CLICKBAIT_DATA/IRISH_TIMES_DATA/irishtimes_data_2018.csv',
'/Users/rossmccrann/MACHINE LEARNING/Group_Project/ml_finale/ml_project/NON_CLICKBAIT_DATA/IRISH_TIMES_DATA/irishtimes_data_2019.csv',
'/Users/rossmccrann/MACHINE LEARNING/Group_Project/ml_finale/ml_project/NON_CLICKBAIT_DATA/IRISH_TIMES_DATA/irishtimes_data_2020.csv',
]

clickbait = [
    "data/buzzfeed_2020.csv",
    "data/boredpanda_2020.csv",
    "data/joedotie_2020.csv",
    "data/theodyssey_2019-20.csv",
    "data/upworthy_2020.csv",
    "data/clickbait_data.csv"]

non_clickbait = [
    "NON_CLICKBAIT_DATA/BBC_DATA/BBC_merged_2010-2020.csv",
    "NON_CLICKBAIT_DATA/CNN_DATA/CNN_merged_2010-2020.csv",
    "NON_CLICKBAIT_DATA/GUARDIAN_DATA/guardian_merged_2010-2020.csv",
    "NON_CLICKBAIT_DATA/IRISH_TIMES_DATA/Irish_times_merged_2010-2020.csv",
    "NON_CLICKBAIT_DATA/NY_TIMES_DATA/NY_Times_merged_2010-2020.csv"]


data_frames = []
'''
for path in clickbait:
    curr_df = pd.read_csv(path, quotechar='"', skipinitialspace=True, dtype='string')
    curr_df.insert(0, 'class', 1)  # label = 1 clickbait
    data_frames.append(curr_df)

for path in non_clickbait:
    curr_df = pd.read_csv(path, quotechar='"', skipinitialspace=True, dtype='string')
    curr_df.insert(0, 'class', 0)  # label = 0 non-clickbait
    data_frames.append(curr_df)
'''

for path in yearly:
    curr_df = pd.read_csv(path, quotechar='"', skipinitialspace=True, dtype='string')
    curr_df.insert(0, 'class', 0)  # label = 1 clickbait
    data_frames.append(curr_df)

i = 0
for df in data_frames:
    df.columns = ['class', 'headline']

    # tokens for POS tagging and lemmatisation later
    df['text'] = tokenize(df['headline'])
    df['text'] = df['text'].apply(remove_punct_tokens)

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

    # lemmatisation
    df['text'] = df['text'].apply(pos_tagging)
    df['text'] = df['text'].apply(change_pos_tag)
    df['text'] = df['text'].apply(lemmatise)
    df['text'] = df['text'].apply(lambda x: [token.lower() for token in x])
    df['text'] = df['text'].apply(remove_stopwords)

    df.drop(columns=['headline'], inplace=True)
    print(df.head())

    df.to_csv(path_or_buf=yearly[i])
    i += 1

#the_motherload = pd.concat(data_frames, ignore_index=True)
#the_motherload.to_csv(path_or_buf="data/features.csv")
