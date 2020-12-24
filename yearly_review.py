import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


def identity_tokenizer(text):
    return text


def print_results(preds, y_vals, title):
    accuracy = accuracy_score(y_vals, preds)
    print("===" + title + "===")
    print(f"Confusion Matrix:\n{confusion_matrix(y_vals, preds)}"
          f"\nAccuracy:{accuracy}")
    print("=== END OF" + title + "===\n\n")
    return accuracy


def yearly_review():
    print('In Yearly')

    results = []

    years = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020]
    for year in years:
        df1 = pd.read_csv(normalised_csv, index_col=0)
        df2 = pd.read_csv('Years/' + str(year) + '.csv')

        # remove current year from the dataset for training, use only the curr year as test data
        pd.concat([df1, df2, df2]).drop_duplicates(keep=False)

        df1["text"] = df1["text"].apply(eval)
        df2["text"] = df2["text"].apply(eval)

        big_features = df1.drop(columns='class')
        big_y = df1['class'].to_numpy()
        print(big_y.shape)
        year_features = df2.drop(columns='class')
        year_y = df2['class'].to_numpy()
        print(year_y.shape)

        tfidf = TfidfVectorizer(tokenizer=identity_tokenizer, ngram_range=(1, 2), lowercase=False)
        tfidf_text = tfidf.fit_transform(big_features['text'])
        big_ef = big_features.drop(columns='text')
        big_X = sparse.hstack([big_ef, tfidf_text]).tocsr()

        tfidf_text = tfidf.transform(year_features['text'])
        little_ef = year_features.drop(columns='text')
        year_X = sparse.hstack([little_ef, tfidf_text]).tocsr()

        ch2 = SelectKBest(chi2, k=10000)
        big_X = ch2.fit_transform(big_X, big_y)
        year_X = ch2.fit_transform(year_X, year_y)

        log_clf = LogisticRegression(C=10, class_weight='balanced', solver='liblinear')
        log_clf.fit(big_X, big_y)

        preds = log_clf.predict(year_X)
        year_accuracy = print_results(preds, year_y, str(year))
        results.append(year_accuracy)

    plt.plot(years, results, 'g^')
    plt.xlabel('year')
    plt.ylabel('accuracy')
    plt.title('Clickbait Yearly Analysis')

#################################

normalised_csv = "data/normalised.csv"
yearly_review()
