import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import seaborn as sns
from scipy import sparse
import math
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, chi2


def identity_tokenizer(text):
    return text


def print_results(preds, y_vals, title):
    print("===" + title + "===")
    print(f"Confusion Matrix:\n{confusion_matrix(y_vals, preds)}"
          f"\nAccuracy:{accuracy_score(y_vals, preds)}"
          f"\nRecall:{recall_score(y_vals, preds)}"
          f"\nF1:{f1_score(y_vals, preds)}"
          f"\nPrecision:{precision_score(y_vals, preds)}")


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
print(np.array(np.unique(y, return_counts=True)).T)
print(y.shape)

tfidf = TfidfVectorizer(tokenizer=identity_tokenizer, ngram_range=(1, 2), lowercase=False)
tfidf_text = tfidf.fit_transform(features['text'])
X_ef = features.drop(columns='text')
X = sparse.hstack([X_ef, tfidf_text]).tocsr()
print(X.shape)
print(y.shape)

# change k for sake of SVC and KNN model runtimes
ch2 = SelectKBest(chi2, k=250)
X = ch2.fit_transform(X, y)
print(X.shape)
print(y.shape)

#################################
# visualising data
# (1) histogram
"""
values = df2.iloc[:, 3:]

values.hist(bins=15, color='steelblue', edgecolor='black', linewidth=1.0, grid=False)
# (2) heatmap
f, ax = plt.subplots(figsize=(10, 6))
corr = values.corr()
hm = sns.heatmap(corr, cmap="coolwarm", annot=True, linewidths=.05, fmt='.2f')
f.suptitle('Clickbait Attributes Correlation Heatmap')
plt.show()
"""

def k_fold_cross_val(k, model, input):
    print(f"=== KFOLD k={k} ===")
    k_fold = KFold(n_splits=k, shuffle=True)
    sq_errs = []
    for train, test in k_fold.split(input):
        model.fit(input[train], y[train])
        print("Model:" + type(model).__name__)
        ypred = model.predict(input[test])
        sq_errs.append(mean_squared_error(y[test], ypred))
        print("Train: " + str(model.score(input[train], y[train])))
        print("Test: " + str(model.score(input[test], y[test])))
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


def plot_top_features(classifier):
    model_coefs = pd.DataFrame(classifier.coef_)
    coefs_df = model_coefs.T
    feature_name_list = list(X_ef.columns)

    # list w/ eng'd features & tf-idf n-grams
    all_feat_names = []
    for i in feature_name_list:
        all_feat_names.append(i)

    for i in tfidf.get_feature_names():
        all_feat_names.append(i)

    # creating column for feat names
    coefs_df['feats'] = all_feat_names
    coefs_df.set_index('feats', inplace=True)
    coefs_df['feats'] = all_feat_names
    coefs_df.set_index('feats', inplace=True)

    # plot non-cb
    coefs_df[0].sort_values(ascending=True).head(20).plot(kind='bar')
    plt.title("SVM: Top 20 Non-Clickbait Coefs")
    plt.xlabel("features")
    plt.ylabel("coef value")
    plt.xticks(rotation=55)
    plt.show()

    # plot CB classification
    coefs_df[0].sort_values(ascending=False).head(20).plot(kind='bar', color='orange')
    plt.title("SVM: Top 20 Clickbait Coefs")
    plt.xlabel("features")
    plt.ylabel("coef value")
    plt.xticks(rotation=55)
    plt.show()


#################################
# model 1 - Logistic Regression
print("\n=== LOGISTIC REGRESSION | L2 PENALTY ===")
# Cross Validation for hyperparameter C:
# c_range = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
# means = []
# std_devs = []
# for C in c_range:
#     log_clf = LogisticRegression(C=C, class_weight='balanced', solver='liblinear')
#     res = k_fold_cross_val(5, log_clf, X)
#     means.append(res[0])
#     std_devs.append(res[1])
#
# log_c_vals = np.log10(c_range)
# error_plot(log_c_vals, means, std_devs, 'LogReg: L2 penalty, varying C', 'log10(C)')

# Logistic Regression model with chosen hyper-parameters:
log_clf = LogisticRegression(C=10, class_weight='balanced', solver='liblinear')
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=20, test_size=0.3)
print(X_train.shape)
print(X_test.shape)

# ROC curve plotting
log_clf.fit(X_train, y_train)
prediction = log_clf.predict_proba(X_test)

fpr, tpr, _ = roc_curve(y_test, prediction[:, 1])
auc_score = roc_auc_score(y_test, prediction[:, 1])
print("AUC Score:", auc_score)
plt.plot(fpr, tpr, color='blue', label='Logistic Regression')

# log_clf.fit(X_train, y_train)
# preds_train = log_clf.predict(X_train)
# preds_test = log_clf.predict(X_test)
#
# print_results(preds_train, y_train, "LogReg train")
# print_results(preds_test, y_test, "LogReg test")
#
# plot_top_features(log_clf)

#################################
# model 2: Gaussian kernel SVM (SVC) model
print("\n=== SVC, Gaussian Kernel ===")
# c_range = [0.001, 1, 1000]
# gammas = [1, 2, 5, 8, 10]
# for C in c_range:
#     means = []
#     std_devs = []
#     for g in gammas:
#         rbf_svc = SVC(C=C, kernel='rbf', gamma=g)
#         y = y_train
#         results = k_fold_cross_val(5, rbf_svc, X_train)
#         means.append(results[0])
#         std_devs.append(results[1])
#         print(f"gamma={g}:", rbf_svc.score(X_test, y_test))
#     plt.errorbar(gammas, means, yerr=std_devs, fmt='.', capsize=5, label=C)
#     plt.plot(gammas, means, linestyle=':', linewidth=2)
# plt.ylabel('mean square error')
# plt.xlabel('gamma')
# plt.title('MSE: C=1, varying γ')
# plt.legend(title='C')
# plt.show()

# SVC Model with chosen parameters
svc = SVC(C=1, kernel='rbf', gamma=5, cache_size=1200, probability=True)

svc.fit(X_train, y_train)
prediction = svc.predict_proba(X_test)
fpr, tpr, _ = roc_curve(y_test, prediction[:, 1])
auc_score = roc_auc_score(y_test, prediction[:, 1])
print("AUC Score:", auc_score)
plt.plot(fpr, tpr, color='orange', label='SVC (Gaussian)')

# svc.fit(X_train, y_train)
# preds_train = svc.predict(X_train)
# print_results(preds_train, y_train, "SVC train")
# preds_test = svc.predict(X_test)
# print_results(preds_test, y_test, "SVC test")


#################################
# model 3 - kNN
print("\n=== kNN ===")

## Cross Validation for hyperparameter n_neighbors:
# neighbours = [1, 3, 5, 7, 9]
# means = []
# stds = []
# for n in neighbours:
#     knn_clf = KNeighborsClassifier(n_neighbors=n, weights='uniform')
#     # knn_clf.fit(X, y)
#     res = k_fold_cross_val(5, knn_clf, X)
#     knn_clf.score(X, y)
#     means.append(res[0])
#     stds.append(res[1])
# error_plot(neighbours, means, stds, 'Prediction Error: varying n_neighbors parameters', 'n_neighbors')

## kNN model with chosen hyperparameter:
knn = KNeighborsClassifier(n_neighbors=5, weights='uniform', n_jobs=2)

knn.fit(X_train, y_train)
prediction = knn.predict_proba(X_test)
fpr, tpr, _ = roc_curve(y_test, prediction[:, 1])
auc_score = roc_auc_score(y_test, prediction[:, 1])
print("AUC Score:", auc_score)
plt.plot(fpr, tpr, color='red', label='K-Neighbours')


# knn.fit(X_train, y_train)
# preds_train = knn.predict(X_train)
# preds_test = knn.predict(X_test)
#
# print_results(preds_train, y_train, "KNN train")
# print_results(preds_test, y_test, "KNN test")
#
# models.append(knn)


dummy = DummyClassifier(strategy='most_frequent')
dummy.fit(X_train, y_train)
preds_train = dummy.predict(X_train)
preds_test = dummy.predict(X_test)

print_results(preds_train, y_train, "Dummy train")
print_results(preds_test, y_test, "Dummy test")

matrix = confusion_matrix(y_train, preds_train)

most_freq_fpr = matrix[0][1] / (matrix[0][1] + matrix[0][0])  # FP / (FP + TN)
most_freq_tpr = matrix[1][1] / (matrix[1][1] + matrix[1][0])  # TP / (TP + FN)

plt.plot(most_freq_fpr, most_freq_tpr, label='Most Frequent Clf.', marker='o', linestyle='None')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.plot([0, 1], [0, 1], color='green', linestyle='--')
plt.title('ROC curves for the chosen classifiers')
plt.legend()
plt.show()  # ROC plot
