import numpy as np
import matplotlib.pyplot as plt


def autolabel(rects, ax):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = np.round(rect.get_height(), 2)
        if height <= 0:
            ax.text(rect.get_x() + rect.get_width() / 2., 1.2 * height,
                    height,
                    ha='center', va='bottom', fontsize=14)
        else:
            ax.text(rect.get_x() + rect.get_width() / 2., 1.05 * height,
                    height,
                    ha='center', va='bottom', fontsize=14)


def draw_plots():
    seaborn_coef = (-0.07, 0.03, 0.11, 0.75, -0.41, 0, 0.09, -0.83, -0.02)
    seaborn_intercept = (1.5, 0.37, 0.55, 0.92, 2.73, 1.53, 1.08, 4.74, 1.83)
    sklearn_coef = (-0.02, 0.03, 0.11, 0.89, -0.43, -0.05, 0.03, -0.86, -0.02)
    sklearn_intercept = (1.45, 0.45, 0.54, 0.8, 2.76, 2.25, 1.32, 4.83, 1.95)
    N = 9
    width = 0.35
    ind = np.arange(N)
    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, seaborn_coef, width, color='r')
    rects2 = ax.bar(ind + width, sklearn_coef, width, color='y')
    ax.set_title('Sklearn Vs Seaborn linear regressor - coefficients')
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(('gender', 'age', 'yearsmarried', 'children', 'religiousness', 'education', 'occupation',
                        'rating', 'agemarried'))
    ax.legend((rects1[0], rects2[0]), ('seaborn', 'sklearn'))
    ax.set_ylabel("coefficieent")
    plt.ylim(-1, 1)
    autolabel(rects1, ax)
    autolabel(rects2, ax)
    plt.show()

    fig2, ax2 = plt.subplots()
    rects11 = ax2.bar(ind, seaborn_intercept, width, color='r')
    rects22 = ax2.bar(ind + width, sklearn_intercept, width, color='y')
    ax2.set_title('Sklearn Vs Seaborn linear regressor - intercepts')
    ax2.set_xticks(ind + width / 2)
    ax2.set_xticklabels(('gender', 'age', 'yearsmarried', 'children', 'religiousness', 'education', 'occupation',
                         'rating', 'agemarried'))
    ax2.legend((rects11[0], rects22[0]), ('seaborn', 'sklearn'))
    ax2.set_ylabel("intercept")
    plt.ylim(0, 5.5)
    autolabel(rects11, ax2)
    autolabel(rects22, ax2)
    plt.show()


def compare_regressors_R2():
    # plt.rcParams.update({'font.size':22})
    r2_score = (0.071, 0.076, -0.142, 0.016, 0.081, 0.074, 0.072, 0.06)
    N = 8
    width = 0.35
    ind = np.arange(N)
    fig, ax = plt.subplots()
    rects = ax.bar(ind, r2_score, width, color=['black', 'red', 'green', 'blue', 'cyan', 'burlywood', 'lime', 'purple'])
    ax.set_xticks(ind + width / 4)
    plt.xticks(rotation=90)
    ax.set_title('R^2 scores for different regressors', fontweight='bold', fontsize=20)
    ax.set_ylabel("R^2 score", fontweight='bold', fontsize=14)
    ax.set_xticklabels(('Multi-variate\n linear\n regression',
                        'Multi-variate\n linear\n regression (scaled)',
                        'Multi-variate\n polynomial\n regression (d=2)',
                        'Multi-variate\n polynomial\n regression scaled(d=2)',
                        'Ridge Regression\n (a=100)',
                        'Lasso\n regression',
                        'Linear Regression -\n Lasso feature\n selection',
                        'pca components\n = 4'), fontweight='bold', fontsize=14)
    plt.xticks(rotation=90)
    plt.ylim(-0.2, 0.2)
    autolabel(rects, ax)
    plt.show()


def compare_regressors_RMSE():
    rmse = (3.28, 0.99, 3.64, 1.02, 3.26, 3.28, 3.29, 3.17)
    N = 8
    width = 0.35
    ind = np.arange(N)
    fig, ax = plt.subplots()
    rects = ax.bar(ind, rmse, width, color=['black', 'red', 'green', 'blue', 'cyan', 'burlywood', 'lime'])
    ax.set_title('Root MSE scores for different regressors', fontweight='bold', fontsize=20)
    ax.set_ylabel("Root Mean Square score", fontweight='bold', fontsize=14)
    ax.set_xticklabels(('Multi-variate\n linear regression',
                        'Multi-variate linear\n regression (scaled)',
                        'Multi-variate polynomial\n regression (d=2)',
                        'Multi-variate polynomial\n regression scaled(d=2)',
                        'Ridge Regression\n (a=100)',
                        'Lasso regression',
                        'Linear Regression -\n Lasso feature selection',
                        'pca components\n = 4'), fontweight='bold', fontsize=14)
    ax.set_xticks(ind + width / 4)
    plt.xticks(rotation=90)
    plt.ylim(0, 5)
    autolabel(rects, ax)
    plt.show()


compare_regressors_R2()
compare_regressors_RMSE()
