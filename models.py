import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix

normalised_csv = "data/normalised.csv"
df = pd.read_csv(normalised_csv)
print(df.head())
X = np.array(df.iloc[:, 1:])
y = np.array(df.iloc[:, 0])

#x1 = np.array(df.iloc[:, 1])
#x2 = np.array(df.iloc[:, 2])
#x3 = np.array(df.iloc[:, 3])
#x4 = np.array(df.iloc[:, 4])
#x5 = np.array(df.iloc[:, 5])
#x6 = np.array(df.iloc[:, 6])
#x7 = np.array(df.iloc[:, 7])

def k_fold_cross_val(k, model,input):
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

#################################
# model 1 - Lasso Regression
print("\n=== LASSO | L1 PENALTY ===")

## Cross Validation for hyperparameter C:
lasso_models = []
ci_range = [0.000001,10,100,1000]
means = [];stds = []
for Ci in ci_range:
    lasso = Lasso(alpha=(1/(2*Ci)))
    lasso.fit(X, y)
    res = k_fold_cross_val(5,lasso,X)
    means.append(res[0])
    stds.append(res[1])
    lasso_models.append(lasso)
plt.errorbar(ci_range, means, yerr=stds, fmt='.', capsize=5)
plt.plot(ci_range, means, linestyle=':',linewidth=2)
plt.ylabel('Mean Square Error')
plt.title('Prediction Error: varying C parameters')
plt.xlabel('Ci Range')
plt.show()

## Lasso model with chosen hyperparameter:
lasso = Lasso(alpha=(1/(2*100)))
lasso.fit(X,y)
ypred = lasso.predict(X)
print("")
print(f"C={100},alpha={(1/(2*100))},intercept={lasso.intercept_},coefs={lasso.coef_}")
#print('lasso model:\n', confusion_matrix(y, ypred))  # tn, fp, tp, fn

