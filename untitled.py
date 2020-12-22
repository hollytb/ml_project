## K MEans Clustering 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import seaborn as sns
from scipy import sparse
from sklearn.cluster import KMeans
import math

def identity_tokenizer(text):
    return text


#################################
# data handling
normalised_csv = "data/normalised.csv"
features_csv = "data/features.csv"
df = pd.read_csv(normalised_csv, index_col=0)
df2 = pd.read_csv(features_csv, index_col=0)
df["text"] = df["text"].apply(eval)
print(df.head())
features = df.drop(columns='class')
y = df['class']
print(y.value_counts())

tfidf = TfidfVectorizer(tokenizer=identity_tokenizer, ngram_range=(1, 2), lowercase=False)
tfidf_text = tfidf.fit_transform(features['text'])
X_ef = features.drop(columns='text')
X = sparse.hstack([X_ef, tfidf_text]).tocsr()
print(X.shape)
print(y.shape)




#text = pd.read_csv('articles1_1000.csv')
text_content = df['text']
tfidf = TfidfVectorizer(tokenizer=identity_tokenizer, ngram_range=(1, 2), lowercase=False).fit_transform(text_content).toarray()
K = range(5, 50, 5)
SSE_mean = []; SSE_std=[]
for k in K:
    gmm = KMeans(n_clusters=k)
    kf = KFold(n_splits=5)
    m=0; v=0
    for train, test in kf.split(tfidf_text):
        gmm.fit(train.reshape(-1, 1))
        cost=-gmm.score(test.reshape(-1, 1))
        m=m+cost; v=v+cost*cost
    SSE_mean.append(m/5); SSE_std.append(math.sqrt(v/5-(m/5)*(m/5)))
plt.errorbar(K, SSE_mean, yerr=SSE_std, xerr=None, fmt='bx-')
plt.ylabel('cost'); plt.xlabel('number of clusters'); plt.show()

k = 15
gmm = KMeans(n_clusters=k).fit(tfidf)
centers = gmm.cluster_centers_.argsort()[:,::-1]; terms = vector.get_feature_names()
for i in range(0,k):
    word_list=[]
    for j in centers[i,:25]:
        word_list.append(terms[j])
    print("cluster%d:"% i); print(word_list)

labels = gmm.predict(tfidf); count = 0
print('\nsimiliar articles:')
for j in range(0,labels.shape[0]):
    if labels[j]==0:
        print('\n'+df['text'].iloc[j])
        count = count+1
        if (count>=5):
            break
