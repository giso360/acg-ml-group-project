from numpy import vectorize
from past.builtins import raw_input
from pygments.lexer import include
from seaborn import axes_style

import processRB as prb
import affais_util as util
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report



df_PT = pd.read_csv("AffairsPT.csv")

X = df_PT.drop(["affairs"], axis=1)
y = df_PT[["affairs"]]

X.replace(["male"], 0, inplace=True)
X.replace(["female"], 1, inplace=True)
X.replace(["no"], 0, inplace=True)
X.replace(["yes"], 1, inplace=True)

y["affairs"][y["affairs"] != 0] = 1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=123)
#
# model = LogisticRegression()
#
# y_pred = model.predict(X_test)
# model.predict(X_test)
# print("accuracy", model.score(X_test, y_test))
# print(classification_report(y_test, y_pred))
#
# test_one = X_test.head(7)
# y_pred_one = model.predict(test_one)
targets = ["faithfull", "cheaters"]
y_pred = util.perform_logistic_regression(X_train, X_test, y_train, y_test, targets)
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, y_pred)
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontweight='bold', fontsize=14)
plt.ylabel('True Positive Rate', fontweight='bold', fontsize=14)
plt.title('Receiver operating characteristic', fontweight='bold', fontsize=20)
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()

# model.score(X_test, y_test)
