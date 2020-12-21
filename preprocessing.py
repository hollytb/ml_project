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


# reading in CLICKBAIT csv files
buzzfeed = "data/buzzfeed_2020.csv"
boredpanda = "data/boredpanda_2020.csv"
joedotie = "data/joedotie_2020.csv"
theodyssey = "data/theodyssey_2019-20.csv"
upworthy = "data/upworthy_2020.csv"

buzzfeed_df = pd.read_csv(buzzfeed, quotechar='"', skipinitialspace=True, dtype='string')
boredpanda_df = pd.read_csv(boredpanda, quotechar='"', skipinitialspace=True, dtype='string')
joedotie_df = pd.read_csv(joedotie, quotechar='"', skipinitialspace=True, dtype='string')
theodyssey_df = pd.read_csv(theodyssey, quotechar='"', skipinitialspace=True, dtype='string')
upworthy_df = pd.read_csv(upworthy, quotechar='"', skipinitialspace=True, dtype='string')

data_frames = [buzzfeed_df,
               boredpanda_df,
               joedotie_df,
               theodyssey_df,
               upworthy_df]

for df in data_frames:
    df.columns = ['headline', 'date']

    # tokens for POS tagging and lemmatisation later
    df['text'] = df['headline'].apply(remove_punctuation)
    df['text'] = tokenize(df['text'])

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

    df.drop(columns=['date', 'headline'], inplace=True)
    print(df.head())

clickbait_df = pd.concat(data_frames, ignore_index=True)

# put non clickbait here too!

clickbait_df.to_csv(path_or_buf="data/clickbait.csv")
