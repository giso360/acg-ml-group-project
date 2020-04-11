from numpy import vectorize
from past.builtins import raw_input
from pygments.lexer import include
# import plot_test as ptst
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

# create csv file with headers from bibliography (https://fairmodel.econ.yale.edu/vote2012/affairs.txt) for redbook
# survey
prb.create_affairs_rb_csv()

print("====================")
print("1.   Load Data\n")
# Psychology Today Dataset
print("Loading dataframe for Psychology Today (PT) dataset from file: AffairsPT.csv\n")
df_PT = pd.read_csv("AffairsPT.csv")
df_PT_2 = df_PT
# Redbook Dataset
print("Loading dataframe for Redbook (RB) dataset from file: ", prb.new_file_name, "\n")
df_RB = pd.read_csv(prb.new_file_name)
print("====================")
print()
print("====================")
##################################################################
##################################################################
print("2.   Describe Datasets")
# Psychology Today Dataset
print("\nPT Info:\n")
print(df_PT.info(verbose=True))
print("\nPT Describe:\n")
util.describe_all(df_PT)
# Redbook Dataset
print("\nRB Info:\n")
print(df_RB.info(verbose=True))
print("\nRB Describe:\n")
util.describe_all(df_RB)
print("====================")
print()
print("====================")
##################################################################
##################################################################
print("\n3.   Data Visualization\n")
##################################################################
print("\nA.   Histograms - Countplots \n")
# TODO: Move code for these figures to external file > affairs_util.py
plt.figure(figsize=(25, 25)).suptitle('Raw Data Histograms for PT survey')
plt.subplot(3, 3, 1)
sns.countplot(df_PT['affairs'], color="teal")
plt.subplot(3, 3, 2)
sns.countplot(df_PT['gender'])
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
print("====================")
print()
print("====================")
##################################################################
##################################################################
print("\n4.   Data Pre-processing\n")
##################################################################
print("\nA.   Apply binary label encoding for categorical data 'gender', 'children': \n")
df_PT_2.replace(["male"], 0, inplace=True)
df_PT_2.replace(["female"], 1, inplace=True)
df_PT_2.replace(["no"], 0, inplace=True)
df_PT_2.replace(["yes"], 1, inplace=True)
##################################################################
print("\nB. Feature engineering: add field 'agemarried' (age that one got married) => age - yearsmarried")
print("Assists to remove false data/outliers")
df_PT["agemarried"] = df_PT["age"] - df_PT["yearsmarried"]
columns_before = df_PT.shape[0]
df_PT = df_PT[df_PT['agemarried'] >= 15.0]  # remove people that were married extremely young
columns_after = df_PT.shape[0]
columns_removed = columns_before - columns_after
print("\nColumns removed due to illegitimate age of marriage: ", str(columns_removed), "\n")
print(df_PT.info())
util.describe_all(df_PT)
##################################################################
print("\nC. Seaborn Regression with regression lines \n")
seaborn_regression_data_per_field = util.regression_plots_one_independent_variable(df_PT)
print("\nOne variable study regression using seaborn regplot tool:\n")
print(seaborn_regression_data_per_field)
print()

print()
##################################################################
print("\nD. Draw Correlation Matrix \n")
util.draw_correlation_heatmap(df_PT)
##################################################################
print("\nE.   Remove unique ID column from dataframe: \n")
df_PT = df_PT.drop(["ID"], axis=1)  # remove ID feature
print("\n====================\n")
##################################################################
print("\nF.   split data to test and training sets: \n")
X = df_PT.drop(["affairs"], axis=1)  # dataframe for independent variables
XX = X
y = df_PT[["affairs"]]  # dataframe for dependent variables
yy = y  # create copy of target column for subsequent PCA operations
# use X, Y dataframes to split test and train data (80/20, consistent randomization)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
print("====================")
print()
print("====================")
##################################################################
##################################################################
print("\n5.   Model Generation\n")
parameters_poly = {'degree': [2, 3, 4, 5, 6, 7]}
parameters_regularization = {'alpha': [1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2,
                                       1, 10, 20, 30, 40, 50, 70, 100, 1000, 10000]}
print("\nA.   one variable cascade linear regression analysis\n")
results_one_variable = util.perform_one_variable_regression_linear(X_train, X_test, y_train, y_test)
print()
##################################################################
print("\n====================\n")
print("\nCompare coefficients from sklearn linear regressor and seaborn\n")
# ptst.draw_plots()
print("\n====================\n")
##################################################################
print("\n====================\n")
print("\nB.   Multi-variate linear regression analysis\n")
# Linear Regression all independent
lreg_multi_variate = LinearRegression()
lreg_multi_variate.fit(X_train, y_train)
affairs_predict = lreg_multi_variate.predict(X_test)
# Model evaluation
print("Coefficients: ", lreg_multi_variate.coef_[0])
print(lreg_multi_variate.intercept_)
print("MSE: ", mean_squared_error(y_test, affairs_predict))
print("Root MSE: ", np.sqrt(mean_squared_error(y_test, affairs_predict)))
print("MAE: ", mean_absolute_error(y_test, affairs_predict))
print("r2 score: ", r2_score(y_test, affairs_predict))
print("vif: ", 1 / (1 - r2_score(y_test, affairs_predict)))
##################################################################
print("\nC.    Attempt to scale results prior to multi variate linear regression analysis...\n")

X_scaled = StandardScaler().fit_transform(XX)
y_scaled = StandardScaler().fit_transform(yy)

X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = train_test_split(X_scaled, y_scaled,
                                                                                test_size=0.2, random_state=123)
# Linear Regression all independent
lreg_multi_variate = LinearRegression()
lreg_multi_variate.fit(X_train_scaled, y_train_scaled)
affairs_predict_scaled = lreg_multi_variate.predict(X_test_scaled)
# Model evaluation
print("Coefficients: ", lreg_multi_variate.coef_[0])
print(lreg_multi_variate.intercept_)
print("MSE: ", mean_squared_error(y_test_scaled, affairs_predict_scaled))
print("Root MSE: ", np.sqrt(mean_squared_error(y_test_scaled, affairs_predict_scaled)))
print("MAE: ", mean_absolute_error(y_test_scaled, affairs_predict_scaled))
print("r2 score: ", r2_score(y_test_scaled, affairs_predict_scaled))
print("vif: ", 1 / (1 - r2_score(y_test_scaled, affairs_predict_scaled)))
# NOTE: small improvement in terms of R2 score
# NOTE: scaling DID NOT improve results for linear multi-variate regression
# NOTE: Future work - Propose Recursive Feature Elimination
##################################################################
print("\nD.    Multi-variate polynomial regression\n")
# Parametric analysis

print("Study on unscaled data")
results_one_variable_polynomial = util.perform_multi_variable_regression_polynomial(X_train, X_test, y_train,
                                                                                    y_test, parameters_poly)

print("Study on scaled data")
results_one_variable_polynomial_scaled = util.perform_multi_variable_regression_polynomial(X_train_scaled,
                                                                                           X_test_scaled,
                                                                                           y_train_scaled,
                                                                                           y_test_scaled,
                                                                                           parameters_poly)

##################################################################
print("\nE.    Ridge regression\n")
ridge_regr = Ridge()
ridge_r = GridSearchCV(ridge_regr, parameters_regularization, scoring='r2', cv=10)
ridge_r.fit(X_train, y_train)
best_alpha = ridge_r.best_params_
print(ridge_r.best_params_)
print(ridge_r.best_score_)
print("Optimal value for alpha hyperparameter for ridge regression is: ", best_alpha.get('alpha'))
ridge_regr = Ridge(alpha=best_alpha.get('alpha'))
ridge_regr = Ridge(alpha=best_alpha.get('alpha'))
ridge_regr.fit(X_train, y_train)
affairs_predict_ridge = ridge_regr.predict(X_test)
# Model evaluation
print("Coefficients: ", ridge_regr.coef_[0])
print(ridge_regr.intercept_)
print("MSE: ", mean_squared_error(y_test, affairs_predict_ridge))
print("Root MSE: ", np.sqrt(mean_squared_error(y_test, affairs_predict_ridge)))
print("MAE: ", mean_absolute_error(y_test, affairs_predict_ridge))
print("r2 score: ", r2_score(y_test, affairs_predict_ridge))
print("vif: ", 1 / (1 - r2_score(y_test, affairs_predict_ridge)))
print("==========================================")
##################################################################
print("\nF.    Lasso regression\n")
lasso_reg = Lasso()
lasso_r = GridSearchCV(lasso_reg, parameters_regularization, scoring='r2', cv=10)
lasso_r.fit(X_train, y_train)
best_alpha = lasso_r.best_params_
print(lasso_r.best_params_)
print(lasso_r.best_score_)
print("Optimal value for alpha hyperparameter for lasso regression is: ", best_alpha.get('alpha'))
lasso_reg = Lasso(alpha=best_alpha.get('alpha'))
lasso_reg = Lasso(alpha=0.01)
lasso_reg.fit(X_train, y_train)
affairs_predict_lasso = lasso_reg.predict(X_test)
print("Coefficients: ", lasso_reg.coef_)
# print("Coefficients: ", lasso_reg.coef_[0])
print(lasso_reg.intercept_)
print("MSE: ", mean_squared_error(y_test, affairs_predict_lasso))
print("Root MSE: ", np.sqrt(mean_squared_error(y_test, affairs_predict_lasso)))
print("MAE: ", mean_absolute_error(y_test, affairs_predict_lasso))
print("r2 score: ", r2_score(y_test, affairs_predict_lasso))
print("vif: ", 1 / (1 - r2_score(y_test, affairs_predict_lasso)))
print("==========================================")
##################################################################
print("\nF.    Multi-variate linear regression with features selected from Lasso regression\n")
X_feature_selection = XX
X_feature_selection = X_feature_selection.drop(["gender", "age", "children"], axis=1)
X_train_selected, X_test_selected, y_train_selected, y_test_selected = train_test_split(X_feature_selection, y,
                                                                                        test_size=0.2, random_state=123)
lreg_multi_variate_selected = LinearRegression()
lreg_multi_variate_selected.fit(X_train_selected, y_train_selected)
affairs_predict_selected = lreg_multi_variate_selected.predict(X_test_selected)
# Model evaluation
print("Coefficients: ", lreg_multi_variate_selected.coef_)
print(lreg_multi_variate_selected.intercept_)
print("MSE: ", mean_squared_error(y_test_selected, affairs_predict_selected))
print("Root MSE: ", np.sqrt(mean_squared_error(y_test_selected, affairs_predict_selected)))
print("MAE: ", mean_absolute_error(y_test_selected, affairs_predict_selected))
print("r2 score: ", r2_score(y_test_selected, affairs_predict_selected))
print("vif: ", 1 / (1 - r2_score(y_test_selected, affairs_predict_selected)))
print("==========================================")
print()
##################################################################
print("\nF.    PCA analysis\n")

# scaled_df = util.perform_pca(df_PT, "affairs", 5)
# X_pca, y_pca = util.drop_keep_column(scaled_df[1], "affairs")
# X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_pca, y_pca,
#                                                                                 test_size=0.2, random_state=123)
# util.perform_multi_variable_regression_linear(X_train_pca, X_test_pca, y_train_pca, y_test_pca)
# pca = PCA(n_components=4)
# X_pca = pca.fit_transform(X_scaled)
# X_pca = pd.DataFrame(data = X_pca, columns=['pc1', 'pc2', 'pc3', 'pc4'])
# affairs_pca_df = pd.concat([X_pca, yy], axis=1)
# print(pca.components_)
# print(pca.explained_variance_)
# print(pca.explained_variance_ratio_)
# print(pca.explained_variance_ratio_.cumsum())
# print("perform multi-variate linear regression with PCA transformed data")
# Y_affairs_pca_df = affairs_pca_df[["affairs"]]
# X_affairs_pca_df = affairs_pca_df.drop(["affairs"], axis=1)
# X_affairs_pca_df = StandardScaler().fit_transform(X_affairs_pca_df)
# Y_affairs_pca_df_2 = StandardScaler(with_mean=False, with_std=False).fit_transform(Y_affairs_pca_df)
# X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_affairs_pca_df, Y_affairs_pca_df,
#                                                                                        test_size=0.2, random_state=123)
# y_test_pca = StandardScaler().fit_transform(y_test_pca)
# y_train_pca = StandardScaler().fit_transform(y_train_pca)
# lreg_multi_variate_pca = LinearRegression()
#
# lreg_multi_variate_pca.fit(X_train_pca, y_train_pca)
# affairs_predict_linear_pca = lreg_multi_variate_pca.predict(X_test_pca)
# Model evaluation
# print("Coefficients: ", affairs_predict_linear_pca.coef_)
# print(affairs_predict_linear_pca.intercept_)
# print("MSE: ", mean_squared_error(y_test_pca, affairs_predict_linear_pca))
# print("Root MSE: ", np.sqrt(mean_squared_error(y_test_pca, affairs_predict_linear_pca)))
# print("MAE: ", mean_absolute_error(y_test_pca, affairs_predict_linear_pca))
# print("r2 score: ", r2_score(y_test_pca, affairs_predict_linear_pca))
# print("vif: ", 1 / (1 - r2_score(y_test_pca, affairs_predict_linear_pca)))
print("==========================================")
