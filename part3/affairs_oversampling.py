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

X, y = drop_keep_column(df_PT, "affairs")

df_PT_minority = df_PT[df_PT.affairs != 0]
df_PT_majority = df_PT[df_PT.affairs == 0]

print(df_PT_majority.shape)
print(df_PT_minority.shape) 
# 451 faithfull Vs 150 other
# Make populate more records from the minority classes to balance the dataset
df_PT_minority_oversampled = resample(df_PT_minority, replace=True,
                                      n_samples=df_PT_majority.shape[0],
                                      random_state=123)

df_PT_new = pd.concat([df_PT_majority, df_PT_minority_oversampled], axis=0).reset_index(drop=True)

plt.figure(figsize=(25, 25)).suptitle('Raw Data Histograms for PT survey')
a = sns.countplot(df_PT_new['affairs'])
util.annotate__bar_plot(a)
plt.show()

# Binary labeling for categorical data
df_PT_new.replace(["male"], 0, inplace=True)
df_PT_new.replace(["female"], 1, inplace=True)
df_PT_new.replace(["no"], 0, inplace=True)
df_PT_new.replace(["yes"], 1, inplace=True)
df_PT_new = df_PT_new.drop(["ID"], axis=1)  # remove ID feature
df_PT_pca = df_PT_new

X = df_PT_new.drop(["affairs"], axis=1)  # dataframe for independent variables
y = df_PT_new[["affairs"]]  # dataframe for dependent variables
XX = X  # create copy of target column for subsequent PCA operations
yy = y  # create copy of target column for subsequent PCA operations

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

print("\n-----------multi-variable linear regression--------------\n")  # R2 -> 11.35%
util.perform_multi_variable_regression_linear(X_train, X_test, y_train, y_test)
print("\n-----------multi-variable polynomial regression--------------\n")  # # R2 -> 20.82%, degree = 3
parameters_poly = {'degree': [2, 3, 4, 5, 6, 7]}
util.perform_multi_variable_regression_polynomial(X_train, X_test, y_train, y_test, parameters_poly)
print("\n--------multi-variable linear regression selected fields-----------------\n")  # R2 - 12.84%
print("\nPerform multi-linear with attrs': \n")
print("\nKeepers: rating, religiousness, yearsmarried, age, children   \n")
df_PT_new = df_PT_new.drop(["gender", "education", "occupation", "age", "children"], axis=1)  # remove other features
X = df_PT_new.drop(["affairs"], axis=1)  # dataframe for independent variables
y = df_PT_new[["affairs"]]  # dataframe for dependent variables
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
util.perform_multi_variable_regression_linear(X_train, X_test, y_train, y_test)

#print("\n--------PCA-----------------\n")  # R2 - 12.84%
#X, y = util.drop_keep_column(df_PT_new, "affairs")
#df_PT_pca.shape
#X_pca, scaled_df = util.perform_pca(df_PT_pca, "affairs", 4)
#y = scaled_df.drop(["affairs"], axis=1)
#X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_pca, y, test_size = 0.2, random_state=42)
#util.perform_multi_variable_regression_linear(X_train_pca, X_test_pca, y_train_pca, y_test_pca)
#print(df_PT_pca.shape)
## majority_count = df_PT[df_PT["affairs"] == 0].shape
## minority_count = df_PT.shape[0] - majority_count[0]
## print("majority records: ", majority_count[0])
## print("minority records: ", minority_count)
#
#a = df_PT["affairs"].unique().tolist
#print(a)

# X = df_PT.drop(["affairs"], axis=1)
# y = df_PT[["affairs"]]
# X.replace(["male"], 0, inplace=True)
# X.replace(["female"], 1, inplace=True)
# X.replace(["no"], 0, inplace=True)
# X.replace(["yes"], 1, inplace=True)
