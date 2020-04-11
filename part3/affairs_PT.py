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


def visualizeDataset():
    plt.figure(figsize=(25, 25)).suptitle('Raw Data Histograms for PT survey')
    plt.subplot(3, 3, 1)
    sns.countplot(df_PT['affairs'], color="teal")
    plt.subplot(3, 3, 2)
    sns.countplot(df_PT['gender'], color="tomato")
    plt.subplot(3, 3, 3)
    sns.distplot(df_PT['age'], color="lime")
    plt.subplot(3, 3, 4)
    sns.distplot(df_PT['yearsmarried'], color="blue")
    plt.subplot(3, 3, 5)
    sns.countplot(df_PT['children'])
    plt.subplot(3, 3, 6)
    sns.countplot(df_PT['religiousness'], color="coral")
    plt.subplot(3, 3, 7)
    sns.countplot(df_PT['education'], color="darksalmon")
    plt.subplot(3, 3, 8)
    sns.countplot(df_PT['occupation'], color="c")
    plt.subplot(3, 3, 9)
    sns.countplot(df_PT['rating'], color="tomato")
    plt.show()


print("\n========= A. DATA LOADING ===========")
print("1.   Load Data\n")
# Psychology Today Dataset
print("Loading dataframe for Psychology Today (PT) dataset from file: AffairsPT.csv\n")
df_PT = pd.read_csv("AffairsPT.csv")

print("\n====================")
print("2.   Data Visualization\n")
visualizeDataset()

print("\n====================")
print("3.   Convert Categorical Data to Numerical Type\n")
df_PT.replace(["male"], 0, inplace=True)
df_PT.replace(["female"], 1, inplace=True)
df_PT.replace(["no"], 0, inplace=True)
df_PT.replace(["yes"], 1, inplace=True)

print("\n========= B. DATA PRE-PROCESSING ===========")
print("3.   Print Dataframe Info to check if there are any missing values\n")
print(df_PT.info())

print("\n====================")
print("4.   Dropping column(s) that serves the purpose of uniquely identifying rows\n")
df_PT = df_PT.drop(["ID"], axis=1)  # remove ID feature

print("\n====================")
print("5.   Feature engineering: add field 'agemarried' (age that one got married) => age - yearsmarried\n")
print("Assists to remove false survey data and outliers")
df_PT["agemarried"] = df_PT["age"] - df_PT["yearsmarried"]
records_before = df_PT.shape[0]
df_PT = df_PT[df_PT['agemarried'] >= 14.0]  # Filter: remove people that were married extremely young
records_after = df_PT.shape[0]
print("records removed after applying the agemarried filter: ", records_before - records_after)

##################################################################
##################################################################
print("\n4.   Data Pre-processing\n")
##################################################################
print("\nA.   Apply binary label encoding for categorical data 'gender', 'children': \n")
df_PT.replace(["male"], 0, inplace=True)
df_PT.replace(["female"], 1, inplace=True)
df_PT.replace(["no"], 0, inplace=True)
df_PT.replace(["yes"], 1, inplace=True)
##################################################################

##################################################################
print("\nC. Seaborn Regression with regression lines \n")
# seaborn_regression_data_per_field = util.regression_plots_one_independent_variable(df_PT)
print("\nOne variable study regression using seaborn regplot tool:\n")
# print(seaborn_regression_data_per_field)
print()

print()
##################################################################
print("\nD. Draw Correlation Matrix \n")
util.draw_correlation_heatmap(df_PT)
##################################################################

for x in range(1, (len(df_PT.columns.tolist()) - 1)):
    print("*********************")
    print("\nno of components: ", x, "\n")
    X_pca, scaled_df, cumExplainedVariance = util.perform_pca(df_PT, "affairs", x)
    print("components = ", x, " | cumulative sum of variance = ", cumExplainedVariance[-1])
    print("*********************")
##################################################################