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





df_PT = pd.read_csv("AffairsPT.csv")

X, y = util.drop_keep_column(df_PT, "affairs")

X = df_PT.drop(["affairs"], axis=1)
y = df_PT[["affairs"]]

X.replace(["male"], 0, inplace=True)
X.replace(["female"], 1, inplace=True)
X.replace(["no"], 0, inplace=True)
X.replace(["yes"], 1, inplace=True)

X_scaled = StandardScaler().fit_transform(X)
pca = PCA(n_components=4)
X_pca = pca.fit_transform(X_scaled)
X_pca = pd.DataFrame(data=X_pca, columns=['pc1', 'pc2', 'pc3', 'pc4'])
# X_pca = pd.DataFrame(data=np.array(X_pca), columns=['pc1', 'pc2', 'pc3', 'pc4', 'pc5', 'pc6'])
scaled_df = pd.concat([X_pca, y], axis=1)
print(pca.explained_variance_ratio_.cumsum())
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_pca, y, test_size=0.2, random_state=42)

lreg_pca = LinearRegression()
lreg_pca.fit(X_train_pca, y_train_pca)
predict_pca = lreg_pca.predict(X_test_pca)
print("r2 score: ", r2_score(y_test_pca, predict_pca))
print("RMSE score: ", np.sqrt(mean_squared_error(y_test_pca, predict_pca)))
