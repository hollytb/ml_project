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

#################################
# visualizing data
# (1) histogram
values = df2.iloc[:, 3:]

values.hist(bins=15, color='steelblue', edgecolor='black', linewidth=1.0, grid=False)
# (2) heatmap
f, ax = plt.subplots(figsize=(10, 6))
corr = values.corr()
hm = sns.heatmap(round(corr, 2), cmap="coolwarm", annot=True, linewidths=.05, fmt='.2f')
f.suptitle('Clickbait Attributes Correlation Heatmap')
plt.show()


def k_fold_cross_val(k, model, input):
    print(f"=== KFOLD k={k} ===")
    k_fold = KFold(n_splits=k, shuffle=True)
    sq_errs = []
    for train, test in k_fold.split(input):
        model.fit(input[train], y[train])
        ypred = model.predict(input[test])
        sq_errs.append(mean_squared_error(y[test], ypred))
    mean = np.mean(sq_errs)
    std = np.std(sq_errs)
    print(f"mean={mean},variance={std}")
    return mean, std


def error_plot(x, means, yerr, title, x_label):
    plt.errorbar(x, means, yerr=yerr, fmt='.', capsize=5)
    plt.plot(x, means, linestyle=':', label='mean', linewidth=2, color='orange')
    plt.legend()
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel('mean square error')
    plt.tight_layout()
    plt.show()


def baseline(x, y, strategy_type):
    dummy = DummyClassifier(strategy=strategy_type)
    dummy.fit(x, y)
    ypred = dummy.predict(x)
    matrix = confusion_matrix(y, ypred)
    accuracy = dummy.score(x, y)
    print("\n=== BASELINE === " + "\nType:" + strategy_type + "\nConfusion Matrix:")
    print(matrix)
    print("Accuracy: " + str(accuracy))
    return matrix


def roc_plot(x, y, models, matrix):
    Xtrain, Xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2)  # polynomial features for LogReg
    for model in models[:2]:
        model.fit(Xtrain, ytrain)
        scores = model.predict_proba(Xtest)
        fpr, tpr, _ = roc_curve(ytest, scores[:, 1])
        print(auc(fpr, tpr))
        model_name = type(model).__name__
        if model_name == 'LogisticRegression':
            model_name = 'LogisticRegression q=2, C=1'
        else:
            model_name = 'kNN Classifier k=3'
        plt.plot(fpr, tpr, label=model_name)
        # Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2)  # switch to normal features for kNN iteration

    """ plotting points of the baseline clf: most_freq """
    most_freq_fpr = matrix[0][1] / (matrix[0][1] + matrix[0][0])  # FP / (FP + TN)
    most_freq_tpr = matrix[1][1] / (matrix[1][1] + matrix[1][0])  # TP / (TP + FN)

    plt.plot(most_freq_fpr, most_freq_tpr, label='Most Frequent Clf.', marker='o', linestyle='None')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.plot([0, 1], [0, 1], color='green', linestyle='--')
    plt.title('ROC curves for the chosen classifiers')
    plt.legend()
    plt.show()


#################################
# model 1 - Logistic Regression
print("\n=== LOGISTIC REGRESSION | L2 PENALTY === \n")

## Cross Validation for hyperparameter C:
models = []
ci_range = [0.001, 1, 10, 100, 1000]
means = [];
stds = []
for Ci in ci_range:
    log_clf = LogisticRegression(penalty='l2', C=Ci, solver="liblinear")
    # log_clf.fit(X, y)
    res = k_fold_cross_val(5, log_clf, X)
    means.append(res[0])
    stds.append(res[1])
error_plot(ci_range, means, stds, 'Prediction Error: varying C parameters', 'Ci Range')

## Logistic Regression model with chosen hyperparameter:
log_clf = LogisticRegression(penalty='l2', C=10, solver="liblinear")
# log_clf = LogisticRegression(penalty='l1',C=10,log_clf = LogisticRegression(penalty='l1',C=10) )
log_clf.fit(X, y)
ypred = log_clf.predict(X)
models.append(log_clf)
matrix = confusion_matrix(y, ypred)
accuracy = log_clf.score(X, y)

print("")
print(f"Hyperparameter C: {10},\nIntercept: {log_clf.intercept_},\nCoefs: {log_clf.coef_}")
print("Accuracy: " + str(accuracy) + "\n" + "Confusion Matrix:" + "\n" + str(matrix))

#################################
# model 2 - kNN
print("\n=== kNN === \n")

## Cross Validation for hyperparameter n_neighbors:
neighbours = [1, 2, 3, 5, 6, 8, 10]
means = [];
stds = []
for n in neighbours:
    knn_clf = KNeighborsClassifier(n_neighbors=n, weights='uniform')
    knn_clf.fit(X, y)
    res = k_fold_cross_val(5, knn_clf, X)
    means.append(res[0])
    stds.append(res[1])
error_plot(neighbours, means, stds, 'Prediction Error: varying n_neighbors parameters', 'n_neighbors')

## kNN model with chosen hyperparameter:
knn_clf = KNeighborsClassifier(n_neighbors=2, weights='uniform')
knn_clf.fit(X, y)
ypred = knn_clf.predict(X)
models.append(knn_clf)
matrix = confusion_matrix(y, ypred)
accuracy = knn_clf.score(X, y)

print("")
print(f"Hyperparameter n: {2},\nIntercept: {log_clf.intercept_},\nCoefs: {log_clf.coef_}")
print("Accuracy: " + str(accuracy) + "\n" + "Confusion Matrix:" + "\n" + str(matrix))

'''
#################################
# model 3 - K-Means
print("\n=== K-Means === \n")

## Cross Validation for hyperparameter k:
K = range(5,50,5)
means = [];stds = []
for k in K:
    gmm = KMeans(n_clusters=k)
    kf = KFold(n_spilts=5)
    m=0;v=0
    for train,test in kf.split(tfidf):
        gmm.fit(train.reshape(-1,1))
        cost =-gmm.score(test.reshape(-1,1))
        m=m+cost; v=v+cost*cost
    means.append(m/5);stds.append(math.sqrt(v/5-(m/5)*(m/5)))
error_plot(K,means,stds,'Prediction Error: varying k parameters','K')

## K-Means model with chosen hyperparameter:
k = 15 # test
gmm = KMeans(n_clusters=k).fit(tfidf)
labels = gmm.predict(tfidf) '''

dummy_martix = baseline(X, y, 'most_frequent')
roc_plot(X, y, models, dummy_martix)
