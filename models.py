import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.linear_model import LogisticRegression

normalised_csv = "data/normalised.csv"
df = pd.read_csv(normalised_csv)
print(df.head())
X = np.array(df.iloc[:, 1:])
y = np.array(df.iloc[:, 0])


# x1 = np.array(df.iloc[:, 1])
# x2 = np.array(df.iloc[:, 2])
# x3 = np.array(df.iloc[:, 3])
# x4 = np.array(df.iloc[:, 4])
# x5 = np.array(df.iloc[:, 5])
# x6 = np.array(df.iloc[:, 6])
# x7 = np.array(df.iloc[:, 7])

def k_fold_cross_val(k, model, input):
    print(f"=== KFOLD k={k} ===")
    k_fold = KFold(n_splits=k)
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


def roc_plot(x, y, models, matrix):
    Xtrain, Xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2)  # polynomial features for LogReg
    for model in models[:1]:
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
ci_range = [0.000001, 1, 10, 100, 1000]
means = [];
stds = []
for Ci in ci_range:
    log_clf = LogisticRegression(penalty='l2', C=Ci)
    log_clf.fit(X, y)
    res = k_fold_cross_val(5, log_clf, X)
    means.append(res[0])
    stds.append(res[1])
error_plot(ci_range, means, stds, 'Prediction Error: varying C parameters', 'Ci Range')

## LogisticRegression model with chosen hyperparameter:
log_clf = LogisticRegression(penalty='l2', C=10)
# log_clf = LogisticRegression(penalty='l1',C=10,log_clf = LogisticRegression(penalty='l1',C=10) )
log_clf.fit(X, y)
ypred = log_clf.predict(X)
models.append(log_clf)
matrix = confusion_matrix(y, ypred)
accuracy = log_clf.score(X, y)

print("")
print(f"Hyperparameter C: {10},\nIntercept: {log_clf.intercept_},\nCoefs: {log_clf.coef_}")
print("Accuracy: " + str(accuracy) + "\n" + "Confusion Matrix:" + "\n" + str(matrix))
roc_plot(X, y, models, matrix)
baseline(X, y, 'most_frequent')
