import processRB as prb
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_fscore_support, f1_score

# TODO: refactor ep_predict

parameters_poly = {'degree': [2, 3, 4, 5, 6, 7]}
parameters_regularization = {'alpha': [1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2,
                                       1, 10, 20, 30, 40, 50, 70, 100, 1000, 10000]}


# def missingValuesDetector(dataframe):
#     fields = []
#     missing = []
#     for col in dataframe.columns:
#         fields.append(col)
#     for field in fields:
#         if()


def regression_plots_one_independent_variable(dataframe):
    """
    create seaborn plots and embed linear regression line with 95% confidence interval shadows

    :param dataframe:
    :return: 2D matrix of the form:
                                    [
                                    [fieldName_1, [coefficient_1, intercept_1],
                                    [fieldName_2, [coefficient_2, intercept_2]]...
                                    ]

    """
    dummy_df = dataframe.select_dtypes(exclude=['object'])
    dummy_df = dummy_df.drop(["ID"], axis=1)
    common_array_of_fields = np.array(dummy_df.columns).tolist()
    i = 0
    color = ["", "teal", "lime", "blue", "coral", "c", "tomato", "darksalmon",
             "silver", "orchid", "cyan"]
    plt.figure(figsize=(25, 25)).suptitle(
        'View Data: Seaborn Regression Plots for PT survey (attributes Vs target) with '
        'embedded regression lines')
    plt.subplots_adjust(hspace=0.5)
    seaborn_regression_data_per_field = []
    sns.set_style("dark")
    for field in common_array_of_fields:
        field_min, field_max, field_range = control_x_range(field, dummy_df)  # To control the x-range of the plot
        copy = np.array(dummy_df.columns).tolist()
        copy.remove("affairs")
        i += 1
        if field == "affairs":
            i = 0
            continue
        else:
            field_regression_data = [field]
            slope_intercept_for_field = []
            copy.remove(field)
            XY_term_PT = dummy_df.drop(copy, axis=1)
            plt.subplot(3, 3, i)
            s = sns.regplot(field, "affairs", data=XY_term_PT, color=color[i],
                            scatter_kws={"alpha": 0.05, "s": 200})
            xd = s.get_lines()[0].get_xdata()
            yd = s.get_lines()[0].get_ydata()
            slope1 = (yd[1] - yd[0]) / (xd[1] - xd[0])
            slope2 = (yd[60] - yd[59]) / (xd[60] - xd[59])
            slope_extreme = (yd[-1] - yd[0]) / (xd[-1] - xd[0])
            intercept1 = yd[0] - slope1 * xd[0]
            intercept2 = yd[0] - slope2 * xd[0]
            intercept_extreme = yd[0] - slope_extreme * xd[0]
            print("xdata for: ", field)
            print(s.get_lines()[0].get_xdata())
            print("ydata for: ", field)
            print("slope for field '", field, "'", " is : ", slope1)
            print("verify slope for field '", field, "'", " is : ", slope2)
            print("Extreme slope for field '", field, "'", " is : ", slope_extreme)
            print("intercept for field '", field, "'", " is : ", intercept1)
            print("verify intercept for field '", field, "'", " is : ", intercept1)
            print("Extreme intercept for field '", field, "'", " is : ", intercept_extreme)
            slope_intercept_for_field.append(slope_extreme)
            slope_intercept_for_field.append(intercept_extreme)
            field_regression_data.append(slope_intercept_for_field)
            seaborn_regression_data_per_field.append(field_regression_data)
            print(s.get_lines()[0].get_ydata())
            sl = np.round(slope_extreme, 2)
            inter = np.round(intercept_extreme, 2)
            text = ""
            # handle 0 coefficient
            if sl == 0:
                text = text + "y ~= " + str(np.round(intercept_extreme, 2)) + " = constant"
            else:
                text = text + "y ~= " + str(np.round(slope_extreme, 2)) + "*" + "x + " + str(
                    np.round(intercept_extreme, 2))
            s.text(field_min, -3, text, fontsize=9)
            plt.xlim(field_min - 0.1 * field_range, field_max + 0.1 * field_range)
            plt.ylim(-5, 15)
    plt.show()

    return seaborn_regression_data_per_field


def control_x_range(field, dataframe):
    field_min = dataframe[field].min()
    field_max = dataframe[field].max()
    field_range = field_max - field_min
    return field_min, field_max, field_range


def describe_all(df):
    with pd.option_context('display.max_columns', 40):
        print(df.describe(include='all'))


def draw_correlation_heatmap(dataframe):
    plt.figure(figsize=(10, 10)).suptitle('Correlation matrix heatmap')
    plt.subplot(2, 1, 1)
    matrix = np.triu(dataframe.corr())
    sns.heatmap(dataframe.corr(), annot=True, mask=matrix, vmin=-1, vmax=1, center=0, cmap='coolwarm')
    plt.yticks(rotation=0)
    plt.xticks(rotation=90)
    plt.show()


# TODO: Create one plot with subplots and add subplot titles and axes names
def perform_one_variable_regression_linear_old(X_train, X_test, y_train, y_test):
    coefficients_one_variable = []
    intercepts_one_variable = []
    R2_one_variable = []
    results_one_variable = []
    for field in X_train.columns:
        lr = LinearRegression()
        X_train_one_var = X_train[field].values.reshape(-1, 1)
        X_test_one_var = X_test[field].values.reshape(-1, 1)
        lr.fit(X_train_one_var, y_train)
        ep_predict_one = lr.predict(X_test_one_var)
        results_one_variable.append(ep_predict_one)
        coefficients_one_variable.append(lr.coef_[0][0])
        intercepts_one_variable.append(lr.intercept_[0])
        R2_one_variable.append(r2_score(y_test, ep_predict_one))
        print("One variable study for predictor: ", field)
        print("==========================================")
        print("Coefficients: ", lr.coef_[0])
        print("Intercept: ", lr.intercept_)
        print("MSE: ", mean_squared_error(y_test, ep_predict_one))
        print("Root MSE: ", np.sqrt(mean_squared_error(y_test, ep_predict_one)))
        print("MEA: ", mean_absolute_error(y_test, ep_predict_one))
        print("R2 score: ", r2_score(y_test, ep_predict_one))
        print("VIF: ", 1 / (1 - r2_score(y_test, ep_predict_one)))
        print()
        print("PLOTS")
        print("::::::::::::::::")
        plt.scatter(X_test[field], y_test, color='black')
        x = np.array(X_test[field])
        y = x * lr.coef_[0] + lr.intercept_
        plt.xlabel(str(field))
        plt.ylabel("affairs")
        plt.plot(x, y, color='blue', linewidth=3)
        plt.show()

    return results_one_variable


def perform_multi_variable_regression_linear(X_train, X_test, y_train, y_test):
    print("\nB.   Multi-variate linear regression analysis\n")
    # Linear Regression all independent
    lreg_multi_variate = LinearRegression()
    lreg_multi_variate.fit(X_train, y_train)
    affairs_predict = lreg_multi_variate.predict(X_test)
    # Model evaluation
    print("Coefficients: ", lreg_multi_variate.coef_[0])
    print("Intercept: ", lreg_multi_variate.intercept_[0])
    print("MSE: ", mean_squared_error(y_test, affairs_predict))
    print("Root MSE: ", np.sqrt(mean_squared_error(y_test, affairs_predict)))
    print("MAE: ", mean_absolute_error(y_test, affairs_predict))
    print("r2 score: ", r2_score(y_test, affairs_predict))
    print("vif: ", 1 / (1 - r2_score(y_test, affairs_predict)))


def perform_multi_variable_regression_polynomial(X_train, X_test, y_train, y_test, parameters):
    """
    pass parameters as => parameters_poly = {'degree': [2, 3, 4, 5, 6, 7]}
    """
    r2_poly = []
    for degree in parameters.get('degree'):
        linear_reg_poly = LinearRegression()
        poly_reg = PolynomialFeatures(degree=degree)
        X_poly_train = poly_reg.fit_transform(X_train)
        X_poly_test = poly_reg.fit_transform(X_test)
        linear_reg_poly.fit(X_poly_train, y_train)
        ep_predict_poly = linear_reg_poly.predict(X_poly_test)
        r2_poly.append(r2_score(y_test, ep_predict_poly))

    optimum_order = np.where(r2_poly == np.max(r2_poly))[0][0] + 2
    print("Optimum polynomial degree: ", optimum_order)
    linear_reg_poly = LinearRegression()
    poly_reg = PolynomialFeatures(degree=optimum_order)
    X_poly_train = poly_reg.fit_transform(X_train)
    X_poly_test = poly_reg.fit_transform(X_test)
    linear_reg_poly.fit(X_poly_train, y_train)
    ep_predict_poly = linear_reg_poly.predict(X_poly_test)
    print("MSE: ", mean_squared_error(y_test, ep_predict_poly))
    print("Root MSE: ", np.sqrt(mean_squared_error(y_test, ep_predict_poly)))
    print("MAE: ", mean_absolute_error(y_test, ep_predict_poly))
    print("r2 score: ", r2_score(y_test, ep_predict_poly))
    print("vif: ", 1 / (1 - r2_score(y_test, ep_predict_poly)))


def scaleDataframe(dataframe, targetName):
    X, y = drop_keep_column(dataframe, targetName)
    components = X.columns.tolist()
    X_scaled = StandardScaler().fit_transform(X)
    X_scaled = pd.DataFrame(data=X_scaled, columns=components)
    scaled_df = pd.concat([X_scaled, y], axis=1)
    return scaled_df


def perform_pca(dataframe, targetName, n_components):
    """
    Purpose: perform PCA analysis on a dataframe gi thee name of target column amd the number of PCA components
    :param X: input dataframe of independent variables
    :param y: input dataframe of dependent variables
    :return: X_pca -> pca transformed dataframe of independent variables
    """
    print("\nPERFORM PCA: START\n")
    X, y = drop_keep_column(dataframe, targetName)
    prefix = "pc"
    components = []
    for component in range(1, n_components + 1):
        components.append(prefix + str(component))
    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    X_pca = pd.DataFrame(data=X_pca, columns=components)
    cumExplainedVariance = pca.explained_variance_ratio_.cumsum()
    scaled_df = pd.concat([X_pca, y], axis=1)
    print("mapping to original components: ")
    print("PCA components as linear combination of original fields:")
    print(pca.components_)
    print("Explained variance:")
    print(pca.explained_variance_)
    print("Explained variance ratio:")
    print(pca.explained_variance_ratio_)
    print("Cumulative sum of explained variance:")
    print(pca.explained_variance_ratio_.cumsum())
    print("PCA columns (indexing with prefix \"pc\"):")
    print(X_pca.columns)
    plt.figure()
    plt.plot(pca.explained_variance_ratio_, '--o', label='explained_variance', color='teal');
    plt.plot(pca.explained_variance_ratio_.cumsum(), '--o', label='cum_explained_variance', color='purple');
    plt.xlabel("Number of components")
    plt.ylabel("explained variance ratio/cumulative explained variance ratio")
    plt.legend()
    plt.show()
    print("\nPERFORM PCA: STOP\n")
    return X_pca, scaled_df, cumExplainedVariance


def annotate__bar_plot(graph):
    for p in graph.patches:
        graph.annotate('{:.0f}'.format(p.get_height()), (p.get_x() + 0.4, p.get_height()),
                       ha='center', va='bottom',
                       color='black')


def drop_keep_column(dataframe, targetName):
    """
    Purpose: for a given dataframe, generate independent and dependent dataframes X, y
    :param dataframe:
    :param targetName:
    :return: X, y from a dataframe
    """
    X = dataframe.drop(targetName, axis=1)
    y = dataframe[[targetName]]
    return X, y


def perform_logistic_regression(X_train, X_test, y_train, y_test, targetNames=None):
    model = LogisticRegression(max_iter=10000)
    model.fit(X_train, y_train.values.ravel())
    y_pred = model.predict(X_test)
    print("accuracy", model.score(X_test, y_test))
    print(classification_report(y_test, y_pred, target_names=targetNames))
    print("F1-macro score is: ", str(np.round(f1_score(y_test, y_pred, average='macro'), 2)))
    return y_pred


def perform_one_variable_regression_linear(X_train, X_test, y_train, y_test):
    coefficients_one_variable = []
    intercepts_one_variable = []
    R2_one_variable = []
    results_one_variable = []
    plt.figure(figsize=(25, 25)).suptitle('One variable scatter Plots for PT survey (attributes Vs target) with '
                                          'embedded regression lines')
    plt.subplots_adjust(hspace=0.5)
    i = 0
    for field in X_train.columns:
        i += 1
        lr = LinearRegression()
        X_train_one_var = X_train[field].values.reshape(-1, 1)
        X_test_one_var = X_test[field].values.reshape(-1, 1)
        lr.fit(X_train_one_var, y_train)
        ep_predict_one = lr.predict(X_test_one_var)
        results_one_variable.append(ep_predict_one)
        coefficients_one_variable.append(lr.coef_[0][0])
        intercepts_one_variable.append(lr.intercept_[0])
        R2_one_variable.append(r2_score(y_test, ep_predict_one))
        print("One variable study for predictor: ", field)
        print("==========================================")
        print("Coefficients: ", lr.coef_[0])
        print("Intercept: ", lr.intercept_)
        print("MSE: ", mean_squared_error(y_test, ep_predict_one))
        print("Root MSE: ", np.sqrt(mean_squared_error(y_test, ep_predict_one)))
        print("MEA: ", mean_absolute_error(y_test, ep_predict_one))
        print("R2 score: ", r2_score(y_test, ep_predict_one))
        print("VIF: ", 1 / (1 - r2_score(y_test, ep_predict_one)))
        print()
        print("PLOTS")
        print("::::::::::::::::")
        plt.subplot(3, 3, i)
        plt.scatter(X_test[field], y_test, color='black')
        x = np.array(X_test[field])
        y = x * lr.coef_[0] + lr.intercept_
        equation = "y = " + str(np.round(lr.coef_[0][0], 2)) + "*x " + "+ " + str(np.round(lr.intercept_[0], 2))
        plt.title(equation)
        plt.xlabel(str(field))
        plt.ylabel("affairs")
        plt.plot(x, y, color='blue', linewidth=3)
    plt.show()

    return results_one_variable
