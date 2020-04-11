from numpy.distutils.system_info import dfftw_info

import processRB as prb
import affais_util as util
import pandas as pd
import seaborn as sns
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

    plt.figure(figsize=(25, 25)).suptitle('Raw Data Histograms for RB survey')
    plt.subplot(3, 3, 1)
    sns.countplot(df_RB['rating'], color="tomato")
    plt.subplot(3, 3, 2)
    sns.distplot(df_RB['age'], color="lime")
    plt.subplot(3, 3, 3)
    sns.distplot(df_RB['yearsmarried'], color="blue")
    plt.subplot(3, 3, 4)
    sns.countplot(df_RB['number_of_children'], color="teal")
    plt.subplot(3, 3, 5)
    sns.countplot(df_RB['religiousness'], color="coral")
    plt.subplot(3, 3, 6)
    sns.countplot(df_RB['education'], color="darksalmon")
    plt.subplot(3, 3, 7)
    sns.countplot(df_RB['occupation'], color="c")
    plt.subplot(3, 3, 8)
    sns.countplot(df_RB['husband_occupation'], color="c")
    plt.subplot(3, 3, 9)
    sns.distplot(df_RB['extramarital_time_per_year'], color="teal")
    plt.show()



print("\n========= A. DATA LOADING ===========")
print("1.   Load Data\n")
# Psychology Today Dataset
print("Loading dataframe for Psychology Today (PT) dataset from file: AffairsRB.csv\n")
df_RB = pd.read_csv(prb.new_file_name)

print("\n====================")
print("2.   Convert Categorical Data to Numerical Type\n")
print(df_RB.info())
print("no categorical values detected")

print("\n====================")
print("3.   Data Visualization\n")
print("Count/Dist plots")
visualizeDataset()

print("\n========= B. DATA PRE-PROCESSING ===========")
print("1.   Print Dataframe Info to check if there are any missing values\n")
# print(df_PT.info())

print("\n====================")
print("2.   Dropping column(s) that serve the purpose of uniquely identifying rows or redundant attributes\n")
df_RB = df_RB.drop(["identifier", "constant",
                    "not_used_1", "not_used_2",
                    "not_used_3"], axis=1)  # remove ID feature

print("\n3.   Convert selected columns from type float64 to int64\n")
df_RB = df_RB.astype({"rating": 'int64', "religiousness": 'int64', "education": 'int64',
                      "occupation": 'int64', "husband_occupation": 'int64'})

print("\n4.   For the field \"number_of_children\" drop records having decimal values as invalid and cast to int64\n")
initial = df_RB.shape[0]
df_RB = df_RB[df_RB["number_of_children"] % 1 == 0]
print("Records dropped (no of children is non-integer value): ", initial - df_RB.shape[0])
df_RB = df_RB.astype({"number_of_children": 'int64'})

print(
    "\n4.   Engineer a new field called \"children\" for boolean interpretation of the \"number_of_children\" field\n")
df_RB = df_RB.astype({"number_of_children": 'int64'})
df_RB["children"] = df_RB["number_of_children"]
df_RB["children"] = df_RB["children"].mask(df_RB["children"] != 0).fillna(1)

print("\n5.   Rename dataframe column \"extramarital_time_per_year\" to \"affairs\" (consistency with affairs_PT)\n")
df_RB = df_RB.rename(columns={"extramarital_time_per_year": "affairs"})

print("\n6.   Create new dataframe column: \"agemarried\"\n")
df_RB["agemarried"] = df_RB["age"] - df_RB["yearsmarried"]
initial = df_RB.shape[0]
df_RB = df_RB[df_RB['agemarried'] >= 14.0]
print("Records dropped (age married is before adolescence): ", initial - df_RB.shape[0])
print("\n====================")

print("\n========= C. FEATURE SELECTION ===========")
print("\n1.   Draw correlation heatmap on unscaled data\n")
util.draw_correlation_heatmap(df_RB)
print("\n2. Draw correlation heatmap on scaled data\n")
util.draw_correlation_heatmap(util.scaleDataframe(df_RB, "affairs"))
print("\n3. Perform PCA analysis\n")

for x in range(1, (len(df_RB.columns.tolist()) - 1)):
    print("*********************")
    print("\nno of components: ", x, "\n")
    X_pca, scaled_df, cumExplainedVariance = util.perform_pca(df_RB, "affairs", x)
    print("components = ", x, " | cumulative sum of variance = ", cumExplainedVariance[-1])
    print("*********************")

print("============ Perform PCA fo n=5 => 80% cumulative explained variance ============")
X_pca, scaled_df, cumExplainedVariance = util.perform_pca(df_RB, "affairs", 5)

print("\n========= D. Analysis ===========")
print("\n1.   univariate linear regression\n")
df_RB = df_RB.drop(["agemarried"], axis=1)  # remove husband_occupation feature
X, y = util.drop_keep_column(df_RB, "affairs")
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                                                test_size=0.2, random_state=123)
util.perform_one_variable_regression_linear(X_train, X_test, y_train, y_test)
print("\n2.   multivariate linear regression\n")
lreg_multi_variate = LinearRegression()
lreg_multi_variate.fit(X_train, y_train)
affairs_predict = lreg_multi_variate.predict(X_test)
print("Coefficients: ", lreg_multi_variate.coef_[0])
print(lreg_multi_variate.intercept_)
print("MSE: ", mean_squared_error(y_test, affairs_predict))
print("Root MSE: ", np.sqrt(mean_squared_error(y_test, affairs_predict)))
print("MAE: ", mean_absolute_error(y_test, affairs_predict))
print("r2 score: ", r2_score(y_test, affairs_predict))
print("vif: ", 1 / (1 - r2_score(y_test, affairs_predict)))
