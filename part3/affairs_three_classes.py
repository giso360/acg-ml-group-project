from functools import partial

from numpy import vectorize
from past.builtins import raw_input
from pygments.lexer import include
from seaborn import axes_style
from sklearn.utils import resample
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


def drop_keep_column(dataframe, columnName):
    return dataframe.drop(columnName, axis=1), dataframe[[columnName]]


df_PT = pd.read_csv("AffairsPT.csv")
class_1 = [1, 2, 3]     # maps to class label 1; MODERATE CHEATERS, affair = 0 => FAITHFULL
class_2 = [7, 12]       # maps to class label 2; EXTREME  CHEATERS
df_PT.replace(["male"], 0, inplace=True)
df_PT.replace(["female"], 1, inplace=True)
df_PT.replace(["no"], 0, inplace=True)
df_PT.replace(["yes"], 1, inplace=True)
df_PT["affairs"][(df_PT["affairs"]).isin(class_1)] = 1
df_PT["affairs"][(df_PT["affairs"]).isin(class_2)] = 2
X, y = drop_keep_column(df_PT, "affairs")

# df_PT_minority = df_PT[df_PT.affairs != 0]
# df_PT_majority = df_PT[df_PT.affairs == 0]

# Make populate more records from the minority classes to balance the dataset
# df_PT_minority_oversampled = resample(df_PT_minority, replace=True,
#                                       n_samples=df_PT_majority.shape[0],
#                                       random_state=123)
#
# df_PT_new = pd.concat([df_PT_majority, df_PT_minority_oversampled], axis=0).reset_index(drop=True)
#

#
# plt.figure(figsize=(25, 25)).suptitle('Raw Data Histograms for PT survey')
# a = sns.countplot(df_PT_new['affairs'])
# util.annotate__bar_plot(a)
# plt.show()

# Binary labeling for categorical data
# df_PT_new.replace(["male"], 0, inplace=True)
# df_PT_new.replace(["female"], 1, inplace=True)
# df_PT_new.replace(["no"], 0, inplace=True)
# df_PT_new.replace(["yes"], 1, inplace=True)
# df_PT_new = df_PT_new.drop(["ID"], axis=1)  # remove ID feature




# Create 3 classes according to affairs unique values:
# 0 - affairs = 0
# 1 - affairs = [1, 2, 3]
# 2 - affairs = [7, 12]

# class_1 = [1, 2, 3]     # maps to class label 1
# class_2 = [7, 12]       # maps to class label 2
#
# df_PT_new["affairs"][(df_PT_new["affairs"]).isin(class_1)] = 1
# df_PT_new["affairs"][(df_PT_new["affairs"]).isin(class_2)] = 2


# X, y = util.drop_keep_column(df_PT_new, "affairs")
# X, y = util.drop_keep_column(df_PT, "affairs")


yy = y  # For subsequent PCA

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

print("\n-------one var linear---------\n")
util.perform_one_variable_regression_linear(X_train, X_test, y_train, y_test)

print("\n-------multi var linear---------\n")
util.perform_multi_variable_regression_linear(X_train, X_test, y_train, y_test)

print("\n-------multi var poly---------\n")
# parameters_poly = {'degree': [2, 3, 4, 5, 6, 7]}
parameters_poly = util.parameters_poly
util.perform_multi_variable_regression_polynomial(X_train, X_test, y_train, y_test, parameters_poly)

print("\n-------logistic poly---------\n")
targets = ["faithfull", "moderate_cheaters", "extreme _cheaters"]
util.perform_logistic_regression(X_train, X_test, y_train, y_test, targets)

