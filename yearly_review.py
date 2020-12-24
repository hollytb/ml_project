import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import seaborn as sns
from scipy import sparse
from sklearn.cluster import KMeans
import math
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectKBest, chi2

def identity_tokenizer(text):
    return text


def print_results(preds, y_vals, title):
    print("===" + title + "===")
    print(f"Confusion Matrix:\n{confusion_matrix(y_vals, preds)}"
          f"\nAccuracy:{accuracy_score(y_vals, preds)}"
          #f"\nRecall:{recall_score(y_vals, preds)}"
          #f"\nF1:{f1_score(y_vals, preds)}"
          #f"\nPrecision:{precision_score(y_vals, preds)}"
          )
    print("=== END OF" + title + "===\n\n")


def YearlyReview(X, y):
    print('In Yearly')
    log_clf = LogisticRegression(C=10, class_weight='balanced', solver='liblinear')
    log_clf.fit(X, y)


    years=[2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020]


    for YEAR in years:

        df = pd.read_csv(normalised_csv, index_col=0)
        # df2 = pd.read_csv(features_csv, index_col=0)
        df["text"] = df["text"].apply(eval)
        print(df.head())
        features = df.drop(columns='class')
        y = df['class'].to_numpy()
        print(y.shape)
        read in year = df1
        read in normalised.csv = df2
        pd.concat([df2,df1,df1]).drop_duplicates(keep=False)

        cy = pd.read_csv('/Users/rossmccrann/MACHINE LEARNING/Group_Project/ml_finale/ml_project/Years/'+ str(YEAR) + '.csv')
        cy["text"] = cy["text"].apply(eval)
        features = cy.drop(columns='class')
        curr_y_year = cy['class']
        print(curr_y_year.value_counts())

        tfidf_text = tfidf.transform(features['text'])
        X_ef = features.drop(columns='text')
        curr_X_year = sparse.hstack([X_ef, tfidf_text]).tocsr()
        
        curr_X_year = ch2.fit_transform(curr_X_year, curr_y_year)
        print(curr_X_year.shape)
        print(curr_y_year.shape)


        preds = log_clf.predict(curr_X_year)
        print_results(preds, curr_y_year, str(YEAR) + ": \n")




#################################
# data handling
normalised_csv = "data/normalised.csv"
features_csv = "data/features.csv"
df = pd.read_csv(normalised_csv, index_col=0)
# df2 = pd.read_csv(features_csv, index_col=0)
df["text"] = df["text"].apply(eval)
print(df.head())
features = df.drop(columns='class')
y = df['class'].to_numpy()
print(y.shape)
tfidf = TfidfVectorizer(tokenizer=identity_tokenizer, ngram_range=(1, 2), lowercase=False)
tfidf_text = tfidf.fit_transform(features['text'])
X_ef = features.drop(columns='text')
X = sparse.hstack([X_ef, tfidf_text]).tocsr()
print(X.shape)
print(y.shape)
ch2 = SelectKBest(chi2, k=20000)
X = ch2.fit_transform(X, y)
print(X.shape)
print(y.shape)
print("Starting Yearly")
YearlyReview(X, y)