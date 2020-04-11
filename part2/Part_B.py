import numpy as np
import pandas as pd
import random
import time
import warnings
from scipy import stats
import scipy.cluster.hierarchy as shc
from sklearn import metrics
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.feature_extraction.image import grid_to_graph
from sklearn.preprocessing import normalize
#from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import metrics


from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.metrics.cluster import normalized_mutual_info_score
import matplotlib.pyplot as plt
#%matplotlib qt
from mpl_toolkits.mplot3d import Axes3D

import seaborn as sns; sns.set(style="ticks",color_codes=True)  # for plot styling
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.set_option('display.max_columns', None)


####################################################################
## Please run the following command for better plot visualization ##
                                                                  ##
#%matplotlib qt                                                   ##
                                                                  ##
####################################################################
                                                                  
#### Loading the data ###

df = pd.read_csv('Wholesale_customers.csv', sep=',')
d= pd.read_csv('Wholesale_customers.csv', sep=',') ##For testing the raw data without preprocessing at the end

########################################
########Data Description ###############
########################################

print("################".center(50, '#'))
print("Data Description".center(50, '*'))
print("################".center(50, '#'))
statistics=df.describe(percentiles=[.05, 0.25, 0.5, 0.75, .95]).T
print("Dataset Shape is:\n", df.shape)

print(statistics)

#numbers of uniques for continuous columns
print()   
print('The number of uniques values in the continuous columns are:'.center(50, '*'))
print()
print('Fresh',df['Fresh'].nunique())
print('Milk',df['Milk'].nunique())
print('Grocery',df['Grocery'].nunique())
print('Frozen',df['Frozen'].nunique())
print('Detergents_Paper',df['Detergents_Paper'].nunique())
print('Delicassen',df['Delicassen'].nunique())
print()

#Print the percentage of data for the nominal columns 

print('The percentage of the nominal columns are:')
print()
print('Channel'.center(50, '*'))
print(df["Channel"].value_counts() / df.shape[0])
print('Region'.center(50, '*'))
print(df["Region"].value_counts() / df.shape[0])
print()


########################################
######### Data  Visualization ##########
########################################

print("################".center(50, '#'))
print("Data Visualisation".center(50, '*'))
print("################".center(50, '#'))
print()
print("Please see pop up figures !")
print()

#Histograms
plt.figure('Raw Data Histograms',figsize=(18,8)).suptitle('Raw Data Histograms')
plt.subplot(2,3,1)
sns.distplot(df['Fresh'],color="teal")
plt.subplot(2,3,2)
sns.distplot(df['Milk'], color="red")
plt.subplot(2,3,3)
sns.distplot(df['Grocery'],color="lime")
plt.subplot(2,3,4)
sns.distplot(df['Frozen'],color="blue")
plt.subplot(2,3,5)
sns.distplot(df['Detergents_Paper'],color="chocolate")
plt.subplot(2,3,6)
sns.distplot(df['Delicassen'],color="coral")
plt.show()

#Clustermap

cor = df.corr()
sns.clustermap(cor,mask=np.zeros_like(cor, dtype=np.bool),cmap=sns.diverging_palette(220, 10, as_cmap=True),robust=True,square=True).fig.suptitle('Raw Data Clustermap')       

#Boxplots
plt.figure('Raw Data Boxplots',figsize=(18,8)).suptitle('Raw Data Boxplots')
plt.subplot(2,3,1)
sns.boxplot(y=df["Fresh"],color="teal")
plt.subplot(2,3,2)
sns.boxplot(y=df["Milk"], color="red")
plt.subplot(2,3,3)
sns.boxplot(y=df["Grocery"],color="lime")
plt.subplot(2,3,4)
sns.boxplot(y=df["Frozen"],color="blue")
plt.subplot(2,3,5)
sns.boxplot(y=df["Detergents_Paper"],color="chocolate")
plt.subplot(2,3,6)
sns.boxplot(y=df["Delicassen"],color="coral")
plt.show()

     
########################################
######### Data  Preprocessing ##########
########################################

print("################".center(50, '#'))
print("Data Preprocessing".center(50, '*'))
print("################".center(50, '#'))
print()

# Drop the categorical values 

df.drop(['Region', 'Channel'], axis = 1, inplace = True)

##Check for Outliers Before
print('Observations before dropping the outliers'.center(50, '*'))
print(df.loc[(df['Fresh']>=15000)&(df['Milk']>=10000)&(df['Grocery']>=15000)].count(axis=0))
    
###Drop Outliers that have more than 2 stant.dev
df=df[(np.abs(stats.zscore(df)) < 2).all(axis=1)]

statistics_new=df.describe(percentiles=[.05, 0.25, 0.5, 0.75, .95])
print(statistics_new)

###Check for Outliers After
    
print('Observations after dropping the outliers'.center(50, '*'))  
print(df.loc[(df['Fresh']>=15000)&(df['Milk']>=10000)&(df['Grocery']>=15000)].count(axis=0))


####Normalize 

df_scaled = normalize(df)
df_scaled = pd.DataFrame(df_scaled, columns=df.columns)


###############################################
### Data  Visualization After Preprocessing ###
###############################################

#Total Pairplot
sns.pairplot(df_scaled,diag_kind='kde').fig.suptitle('Preprocessed Data Pairplot') 

time.sleep(5) #because need some time for the pairplot

#Boxplots
plt.figure('Preprocessed Data Boxplots',figsize=(18,8)).suptitle('Preprocessed Data Boxplots')
plt.subplot(2,3,1)
sns.boxplot(y=df_scaled["Fresh"],color="teal")
plt.subplot(2,3,2)
sns.boxplot(y=df_scaled["Milk"], color="red")
plt.subplot(2,3,3)
sns.boxplot(y=df_scaled["Grocery"],color="lime")
plt.subplot(2,3,4)
sns.boxplot(y=df_scaled["Frozen"],color="blue")
plt.subplot(2,3,5)
sns.boxplot(y=df_scaled["Detergents_Paper"],color="chocolate")
plt.subplot(2,3,6)
sns.boxplot(y=df_scaled["Delicassen"],color="coral")
plt.show()

#Histograms
plt.figure('Preprocessed Data Histograms',figsize=(18,8)).suptitle('Preprocessed Data Histograms')
plt.subplot(2,3,1)
sns.distplot(df_scaled['Fresh'],color="teal")
plt.subplot(2,3,2)
sns.distplot(df_scaled['Milk'], color="red")
plt.subplot(2,3,3)
sns.distplot(df_scaled['Grocery'],color="lime")
plt.subplot(2,3,4)
sns.distplot(df_scaled['Frozen'],color="blue")
plt.subplot(2,3,5)
sns.distplot(df_scaled['Detergents_Paper'],color="chocolate")
plt.subplot(2,3,6)
sns.distplot(df_scaled['Delicassen'],color="coral")
plt.show()

#Clustermap
corr = df_scaled.corr()
sns.clustermap(corr,mask=np.zeros_like(corr, dtype=np.bool),cmap=sns.diverging_palette(220, 10, as_cmap=True),robust=True,square=True).fig.suptitle('Preproced Data Clustermap')       
plt.show()
 

###################################
############### PCA ###############
###################################

## PCA using the method of *Minka, T. P. "Automatic choice of dimensionality for PCA". In NIPS, pp. 598-604*

pca = PCA(n_components='mle')

tr=pca.fit(df_scaled)
X_n = tr.transform(df_scaled)

finaldf = pd.DataFrame(X_n , columns = ['principal component 1', 'principal component 2','principal component 3', 'principal component 4'])
print(pca.explained_variance_ratio_)
print()

########################################
######## Clustering Algorithms #########
########################################
print("=====================================================")
print("               Clustering Algorithms                 ")
print("===================================================== \n")


##############
### DBSCAN ### 
############## 

print("1)      DBSCAN Clustering         ")
print("-------------------------------- \n")
random.seed(1)
DBSCANdf=finaldf.copy()
"""
##########################################################################################
################# Optimizing parameters eps and min_samples for DBSCAN ###################

# n_samples
plt.figure("optimun n_samples for DBCAN")
n_samples = np.arange(2, 100,1)
models = [DBSCAN(eps=0.26, min_samples=n).fit(DBSCANdf) for n in n_samples]
plt.plot(n_samples, [metrics.silhouette_score(DBSCANdf, m.labels_) for m in models],'r-')
plt.suptitle('Optimizing of min_samples DBSCAN - Silhouette')
plt.xticks(np.arange(0, 101,5))
plt.xlabel('number of min_samples')
plt.ylabel('Silhoiuette')

# eps
plt.figure("optimun eps for DBSCAN")
n_eps = np.arange(0.13,1.0,0.01)
models = [DBSCAN(eps=float(n), min_samples=61).fit(DBSCANdf) for n in n_eps]
plt.plot(n_eps, [metrics.silhouette_score(DBSCANdf, m.labels_) for m in models],'g-')
plt.suptitle('Optimizing of eps DBSCAN - Silhouette')
plt.xticks(np.arange(0.1, 1.01,0.1))
plt.yticks(np.arange(0.1,0.710,0.05))
plt.xlabel('number of eps')
plt.ylabel('Silhoiuette')

##########################################################################################   
"""
### Optimal model ### 

print("Optimal model")
print("--------------")
#Fit the data into the model with optimal parameters 
dba = DBSCAN(eps=0.26, min_samples=61)
dba.fit(DBSCANdf)

#Predict the clusters of each datum
dbs= dba.labels_

#Create a new column and assigned the clusters 
TagsDBSCAN=pd.DataFrame(dbs)
DBSCANdf["Cluster_DBSCAN"]=TagsDBSCAN.replace(-1,1)
n_clusters_DBSCAN = len(set(dbs)) 

#Evaluate the model 
print("Silhouette: %0.3f"% metrics.silhouette_score(DBSCANdf, dbs))
print("\nEstimated number of clusters: %d" % n_clusters_DBSCAN)

#Create a dictionary and indexing the data of each cluster
clusters_pca={}
clustersCount_pca={}
for i in range(0,378):
    if dbs[i] in clusters_pca.keys():
        clusters_pca[dbs[i]].append(i)
        clustersCount_pca[dbs[i]]=clustersCount_pca[dbs[i]]+1
    else:
        clusters_pca[dbs[i]]=[i]
        clustersCount_pca[dbs[i]]=1
print('the Clusters distribution on DBSCAN is: ',clustersCount_pca)

###################################
##### DBSCAN clustering plots #####
###################################

### Calculate the cendroids for ploting
#Find Centroids
Dbc = NearestCentroid()

#Optimal Model
DBO_centr=Dbc.fit(DBSCANdf, dbs).centroids_

#Scatterplot 
plt.figure('Scatterplot DBSCAN Optimal')
ax=sns.scatterplot(x=DBSCANdf['principal component 1'], y=DBSCANdf['principal component 2'], hue=DBSCANdf["Cluster_DBSCAN"],alpha=0.75,legend=False)
ax.scatter(x=DBSCANdf['principal component 1'], y=DBSCANdf['principal component 2'],c=DBSCANdf["Cluster_DBSCAN"],cmap='rainbow',alpha=0.9,)
plt.scatter(DBO_centr[:, 0], DBO_centr[:, 1], c='black', s=200, alpha=0.8)
ax.set_title("DBSCAN Clustering with 2 Clusters")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

#boxplot
plt.figure('Barplot DBSCAN Optimal')
sns.boxplot(x='Cluster_DBSCAN', y='principal component 1',notch=True, data=DBSCANdf).set_title("DBSCAN Optimal Boxplot")

### 3D Scatterplot ###
fig = plt.figure('3D DBSCAN',figsize=(20,20))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(DBSCANdf['principal component 1'][DBSCANdf.Cluster_DBSCAN == 0], DBSCANdf['principal component 2'][DBSCANdf.Cluster_DBSCAN == 0], DBSCANdf['principal component 3'][DBSCANdf.Cluster_DBSCAN == 0], c='crimson',alpha=0.8, s=70)
ax.scatter(DBSCANdf['principal component 1'][DBSCANdf.Cluster_DBSCAN == 1], DBSCANdf['principal component 2'][DBSCANdf.Cluster_DBSCAN == 1], DBSCANdf['principal component 3'][DBSCANdf.Cluster_DBSCAN == 1], c='cyan',alpha=1, s=70)
ax.set_title("DBSCAN Clustering 3D")
ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
ax.set_zlabel("Principal Component 3")
plt.show()

print("______________________________________")

##############
### KMEANS ###
##############

print("\n 2)        KMEANS Clustering      ")
print("---------------------------------- \n")
KMEANSdf=finaldf.copy()
random.seed(1)
"""
###################################################################################
################### Optimizing Number of Clusters #################################

# Silhouette (optimun = 2)
plt.figure('optimun_n_of_clusters_Silhouette_KMEANS')
n_clusters = np.arange(2, 21)
models = [KMeans(n_clusters=n,).fit(KMEANSdf) for n in n_clusters]
plt.plot(n_clusters, [metrics.silhouette_score(KMEANSdf, m.predict(KMEANSdf)) for m in models],'r*-')
plt.suptitle('Optimizing of KMEANS - Silhouette')
plt.xticks(np.arange(2, 21,2))
plt.xlabel('number of components')
plt.ylabel('Silhoiuette')

# Inertia (optimun = 20)
plt.figure('optimun_n_of_clusters_Inertia_KMEANS')
n_clusters = np.arange(2, 21)
models = [KMeans(n_clusters=n,).fit(KMEANSdf) for n in n_clusters]
plt.plot(n_clusters,[ m.inertia_ for m in models],'b*-')
plt.suptitle('Optimizing of KMEANS - Inertia')
plt.xticks(np.arange(2, 21,2))
plt.xlabel('number of components')
plt.ylabel('Inertia')


#Visualize the changing of the clusters while the number increasing
for n in range(2,21):
    print ('Clustering for n=',n)
    kmeans = KMeans(n_clusters=n)
    kmeans.fit(KMEANSdf)
    y_kmeans = kmeans.predict(KMEANSdf)
    cn=kmeans.cluster_centers_
    plt.figure('KMEANS_N= %.0f' % n)
    plt.suptitle('Clustering for n = %.0f' % n)
    plt.scatter(cn[:, 0], cn[:, 1], c='red', s=250, alpha=0.9)
    #sns.scatterplot(x=KMEANSdf['principal component 1'], y=KMEANSdf['principal component 2'],color='blue', alpha=0.75,legend=False)
    ax=sns.scatterplot(x=KMEANSdf['principal component 1'], y=KMEANSdf['principal component 2'],hue=y_kmeans,legend=False)
    ax.scatter(KMEANSdf['principal component 1'], y=KMEANSdf['principal component 2'],c=y_kmeans,cmap='viridis',alpha=0.9,)


###################################################################################
print("From the above looks like optimal is the 5-clusters model but we also need the 2-clusters model for comparison")
"""

### Optimal model (n=5) ### 
print("Optimal model:")
print("-----------------")

#Fit the model with optimal, and use the 4 cores to run it faster
km = KMeans(n_clusters=5,n_jobs=4, tol=1e-7, random_state=None)
km.fit(KMEANSdf)
# Predict the clusters of each datum
Kms1=km.predict(KMEANSdf)

#Evaluate the model
print ("Inertia: %0.3f" %km.inertia_)
silhouette_values = metrics.silhouette_score(KMEANSdf, Kms1)
print ("Silhouette: %0.3f" % silhouette_values)
n_clusters_KMEANS = len(set(Kms1)) 
print("\nEstimated number of clusters: %d" % n_clusters_KMEANS)

### Comparison model (n=2) ### For downgrading 
print("\nComparison model: ")
print("-------------------")

#Fit the model with optimal, and use the 4 cores to run it faster
km = KMeans(n_clusters=2,n_jobs=4)
km.fit(KMEANSdf)
# Predict the clusters of each datum
Kms2=km.predict(KMEANSdf)

#Evaluate the model
print ("Inertia: %0.3f" % km.inertia_)
silhouette_values = metrics.silhouette_score(KMEANSdf, Kms2)
print ("Silhouette: %0.3f" % silhouette_values)
print("NMI for downgrading from 5 to 2 cluster is: %0.3f" %normalized_mutual_info_score(Kms1, Kms2))
n_clusters_cKMEANS = len(set(Kms2)) 
print("\nEstimated number of clusters: %d" % n_clusters_cKMEANS)

#Adding the columns to the Dataset
KMEANSdf["Cluster_KMEANS"] = pd.DataFrame(Kms1)
KMEANSdf["Cluster(n=2)_KMEANS"] = pd.DataFrame(Kms2)  

print("NMI: %0.3f" %normalized_mutual_info_score(Kms2, DBSCANdf["Cluster_DBSCAN"]))

 
####################################
#####  KMEANS clustering plots #####
####################################

### Calculate the cendroids for ploting
#Find Centroids
Kmc = NearestCentroid()

#Optimal Model
KMO_centr=Kmc.fit(KMEANSdf, Kms1).centroids_
#Comparison Model
KMC_centr=Kmc.fit(KMEANSdf, Kms2).centroids_
 
### Optimal model (n=5) ### 
#Scatterplot 
plt.figure('Scatterplot KMEANS Optimal')
ax=sns.scatterplot(x=KMEANSdf['principal component 1'], y=KMEANSdf['principal component 2'],hue=KMEANSdf["Cluster_KMEANS"],legend=False)
ax.scatter(KMEANSdf['principal component 1'], y=KMEANSdf['principal component 2'],c=KMEANSdf["Cluster_KMEANS"],cmap='rainbow',alpha=0.9,)
plt.scatter(KMO_centr[:, 0], KMO_centr[:, 1], c='black', s=200, alpha=0.8)
ax.set_title("KMEANS Clustering with 5 Clustes")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
#boxplot
plt.figure('Barplot KMEANS Optimal')
sns.boxplot(x='Cluster_KMEANS', y='principal component 1',notch=True, data=KMEANSdf).set_title("KMEANS Optimal Boxplot")

### Comparison model (n=2) ###
#Scatterplot 
plt.figure('Scatterplot KMEANS for Comparison')
plt.scatter(KMC_centr[:, 0], KMC_centr[:, 1], c='black', s=200, alpha=1)
ax=sns.scatterplot(x=KMEANSdf['principal component 1'], y=KMEANSdf['principal component 2'],hue=KMEANSdf["Cluster(n=2)_KMEANS"],cmap='viridis',legend=False)
ax.scatter(KMEANSdf['principal component 1'], y=KMEANSdf['principal component 2'],c=KMEANSdf["Cluster(n=2)_KMEANS"],cmap='rainbow')
ax.set_title("KMEANS Clustering with 2 Clusters for Comparison")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
#boxplot
plt.figure('Barplot KMEANS for Comparison')
sns.boxplot(x='Cluster(n=2)_KMEANS', y='principal component 1',notch=True, data=KMEANSdf).set_title("KMEANS Boxplot for Comparison")


###### 3D plot #######

fig = plt.figure('3D KMEANS',figsize=(20,20))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(KMEANSdf['principal component 1'][KMEANSdf.Cluster_KMEANS == 0], KMEANSdf['principal component 2'][KMEANSdf.Cluster_KMEANS == 0], KMEANSdf['principal component 3'][KMEANSdf.Cluster_KMEANS == 0], c='blue',alpha=0.8, s=70)
ax.scatter(KMEANSdf['principal component 1'][KMEANSdf.Cluster_KMEANS == 1], KMEANSdf['principal component 2'][KMEANSdf.Cluster_KMEANS == 1], KMEANSdf['principal component 3'][KMEANSdf.Cluster_KMEANS == 1], c='lime',alpha=1, s=70)
ax.scatter(KMEANSdf['principal component 1'][KMEANSdf.Cluster_KMEANS == 2], KMEANSdf['principal component 2'][KMEANSdf.Cluster_KMEANS == 2], KMEANSdf['principal component 3'][KMEANSdf.Cluster_KMEANS == 2], c='fuchsia',alpha=0.8, s=70)
ax.scatter(KMEANSdf['principal component 1'][KMEANSdf.Cluster_KMEANS == 3], KMEANSdf['principal component 2'][KMEANSdf.Cluster_KMEANS == 3], KMEANSdf['principal component 3'][KMEANSdf.Cluster_KMEANS == 3], c='red',alpha=1, s=70)
ax.scatter(KMEANSdf['principal component 1'][KMEANSdf.Cluster_KMEANS == 4], KMEANSdf['principal component 2'][KMEANSdf.Cluster_KMEANS == 4], KMEANSdf['principal component 3'][KMEANSdf.Cluster_KMEANS == 4], c='black',alpha=1, s=70)
ax.view_init(30, 185)
ax.set_title("KMEANS 3D Clustering")
ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
ax.set_zlabel("Principal Component 3")
plt.show()
print("______________________________________")


##########################
### Gaussian Mixtures ####
##########################

print("\n 3)   Gausiann Mixtures Clustering       ")
print("---------------------------------------- \n")
Gaussiandf=finaldf.copy()
random.seed(1)
"""
############################################################################################
###################### Optimizing Number of Components #####################################

# Silhouette (optimun = 2)
plt.figure('optimun_n_of_clusters_Silhouette_GaussianMixture')
n_components = np.arange(2, 21)
models = [GaussianMixture(n, covariance_type='full', random_state=0).fit(Gaussiandf) for n in n_components]
plt.plot(n_components, [metrics.silhouette_score(Gaussiandf, m.predict(Gaussiandf)) for m in models],'r*-')
plt.suptitle('Optimizing of Gaussian Mixture - Silhouette')
plt.xticks(np.arange(2, 21,2))
plt.xlabel('number of components')
plt.ylabel('Silhoiuette')

# Likehood (optimun = 17)
plt.figure('optimun_n_of_clusters_Likehood_GaussianMixture')
n_components = np.arange(2, 21)
models = [GaussianMixture(n, covariance_type='full', random_state=0).fit(Gaussiandf) for n in n_components]
plt.plot(n_components, [m.score(Gaussiandf) for m in models],'g*-')
plt.suptitle('Optimizing of Gaussian Mixture - Likehood')
plt.xticks(np.arange(2, 21,2))
plt.xlabel('number of components')
plt.ylabel('Likehood')
 

# BIC(Bayesian Information Criterion) (optimun = 6)
# AIC(Akaike Information Criterion) (optimun = 17)
plt.figure('optimun_n_of_clusters_AIC/BIC_GaussianMixture')
n_components = np.arange(2, 21)
models = [GaussianMixture(n, covariance_type='full', random_state=0).fit(Gaussiandf) for n in n_components]
plt.plot(n_components, [m.bic(Gaussiandf) for m in models], 'b*-',label='BIC')
plt.plot(n_components, [m.aic(Gaussiandf) for m in models],'y*-', label='AIC')
plt.legend(loc='best')
plt.xlabel('n_components')
plt.suptitle('Optimizing of Gaussian Mixture - BIC & AIC')
############################################################################################ 
print("Compairing all the above metrics the optimal model is with 6 components")
print("Also we keep the 2 components for comparison")
"""

### Optimal model (n=6) ###
print("Optimal model:")
print("-----------------")

#train the optimal model, using warm start for faster and more acurrate results
gmm = GaussianMixture(n_components=6, covariance_type='full', verbose=2,warm_start=True,max_iter=1000)
gmm.fit(Gaussiandf)
#Predict the componet of each datum
gm_y1 = gmm.predict(Gaussiandf)
metrics.silhouette_score(Gaussiandf, gm_y1)
#Evaluate the model
print ("Silhouette: %0.3f" % silhouette_values)
print ("Likehood: %0.3f" %gmm.score(Gaussiandf))
bic=GaussianMixture(n_components=6, covariance_type='full').fit(Gaussiandf).bic(Gaussiandf)
print("Bayesian Information Criterion: %0.3f" % bic)
aic=GaussianMixture(n_components=6, covariance_type='full').fit(Gaussiandf).aic(Gaussiandf)
print("Akaike Information Criterion: %0.3f" % aic)
n_clusters_GM = len(set(gm_y1)) 
print("\nEstimated number of Components: %d" % n_clusters_GM)
#gmm.get_params(deep=True)

### Comparison model (n=2) ###
print("\nComparison model: ")
print("-------------------")

#train the comparison model
gmm = GaussianMixture(n_components=2, covariance_type='full', verbose=2,warm_start=True,max_iter=1000)
gmm.fit(Gaussiandf)
#Predict the componet of each datum
gm_y2 = gmm.predict(Gaussiandf)
#Evaluate the model
metrics.silhouette_score(Gaussiandf, gm_y2)
print ("Silhouette: %0.3f" % silhouette_values)
print ("Likehood: %0.3f" %gmm.score(Gaussiandf))
bic=GaussianMixture(n_components=2, covariance_type='full').fit(Gaussiandf).bic(Gaussiandf)
print("Bayesian Information Criterion: %0.3f" % bic)
aic=GaussianMixture(n_components=2, covariance_type='full').fit(Gaussiandf).aic(Gaussiandf)
print("Akaike Information Criterion: %0.3f" % aic)
print("NMI for downgrading from 6 to 2 componets is: %0.3f" %normalized_mutual_info_score(gm_y1, gm_y2))
n_clusters_cGM = len(set(gm_y2)) 
print("\nEstimated number of Components: %d" % n_clusters_cGM)

#Adding the columns to the Dataset
Gaussiandf['Cluster_Gaussian']=pd.DataFrame(gm_y1)
Gaussiandf['Cluster(n=2)_Gaussian']=pd.DataFrame(gm_y2)

####################################
#####  Gaussian Mixtures plots #####
####################################

### Calculate the cendroids for ploting
#Find Centroids
Gmc = NearestCentroid()

#Optimal Model
GMO_centr=Gmc.fit(Gaussiandf, gm_y1).centroids_
#Comparison Model
GMC_centr=Gmc.fit(Gaussiandf, gm_y2).centroids_


### Optimal model (n=6) ### 
#Scatterplot 
plt.figure('Scatterplot Gaussian Mixtures Optimal')
ax=sns.scatterplot(x=Gaussiandf['principal component 1'], y=Gaussiandf['principal component 2'],hue=Gaussiandf["Cluster_Gaussian"],legend=False)
ax.scatter(Gaussiandf['principal component 1'], y=Gaussiandf['principal component 2'],c=Gaussiandf["Cluster_Gaussian"],cmap='rainbow',alpha=0.9,)
plt.scatter(GMO_centr[:, 0], GMO_centr[:, 1], c='black', s=200, alpha=0.8)
ax.set_title("Gaussian Mixtures with 6 Components")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
#boxplot
plt.figure('Barplot Gaussian Mixtures Optimal')
sns.boxplot(x='Cluster_Gaussian', y='principal component 1',notch=True, data=Gaussiandf).set_title("Gaussian Mixture Optimal Boxplot")


### Comparison model (n=2) ###
#Scatterplot 
plt.figure('Scatterplot Gaussian Mixtures for Comparison')
ax=sns.scatterplot(x=Gaussiandf['principal component 1'], y=Gaussiandf['principal component 2'],hue=Gaussiandf["Cluster(n=2)_Gaussian"],legend=False)
ax.scatter(Gaussiandf['principal component 1'], y=Gaussiandf['principal component 2'],c=Gaussiandf["Cluster(n=2)_Gaussian"],cmap='rainbow',alpha=0.9,)
plt.scatter(GMC_centr[:, 0], GMC_centr[:, 1], c='black', s=200, alpha=0.8)
ax.set_title("Gaussian Mixtures with 2 Components for Comparison")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
#boxplot
plt.figure('Barplot Gaussian Mixtures for Comparison')
sns.boxplot(x='Cluster(n=2)_Gaussian', y='principal component 1',notch=True, data=Gaussiandf).set_title("Gaussian Mixture Boxplot for Comparison")

###### 3D plot #######

fig = plt.figure('3D Gaussian Mixtures',figsize=(20,20))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(Gaussiandf['principal component 1'][Gaussiandf.Cluster_Gaussian == 0], Gaussiandf['principal component 2'][Gaussiandf.Cluster_Gaussian == 0], Gaussiandf['principal component 3'][Gaussiandf.Cluster_Gaussian == 0], c='blue',alpha=0.8, s=70)
ax.scatter(Gaussiandf['principal component 1'][Gaussiandf.Cluster_Gaussian == 1], Gaussiandf['principal component 2'][Gaussiandf.Cluster_Gaussian == 1], Gaussiandf['principal component 3'][Gaussiandf.Cluster_Gaussian == 1], c='cyan',alpha=1, s=70)
ax.scatter(Gaussiandf['principal component 1'][Gaussiandf.Cluster_Gaussian == 2], Gaussiandf['principal component 2'][Gaussiandf.Cluster_Gaussian == 2], Gaussiandf['principal component 3'][Gaussiandf.Cluster_Gaussian == 2], c='purple',alpha=0.8, s=70)
ax.scatter(Gaussiandf['principal component 1'][Gaussiandf.Cluster_Gaussian == 3], Gaussiandf['principal component 2'][Gaussiandf.Cluster_Gaussian == 3], Gaussiandf['principal component 3'][Gaussiandf.Cluster_Gaussian == 3], c='red',alpha=1, s=70)
ax.scatter(Gaussiandf['principal component 1'][Gaussiandf.Cluster_Gaussian == 4], Gaussiandf['principal component 2'][Gaussiandf.Cluster_Gaussian == 4], Gaussiandf['principal component 3'][Gaussiandf.Cluster_Gaussian == 4], c='black',alpha=1, s=70)
ax.scatter(Gaussiandf['principal component 1'][Gaussiandf.Cluster_Gaussian == 5], Gaussiandf['principal component 2'][Gaussiandf.Cluster_Gaussian == 5], Gaussiandf['principal component 3'][Gaussiandf.Cluster_Gaussian == 5], c='lime',alpha=1, s=70)
ax.view_init(30, 185)
ax.set_title("Gaussian Mixture 3D")
ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
ax.set_zlabel("Principal Component 3")
plt.show()

print("______________________________________")


#################################
### Agglomerative Clustering ####
#################################

print("\n 4)   Agglomerative Clustering       ")
print("---------------------------------------- \n")
Hierdf=finaldf.copy()
random.seed(1)

"""

#Plot a matrix dataset as a hierarchically-clustered heatmap
corl = df_scaled.corr()
sns.clustermap(corl,mask=np.zeros_like(corl, dtype=np.bool),cmap=sns.diverging_palette(220, 10, as_cmap=True),robust=True,square=True)  
corl = Hierdf.corr()
sns.clustermap(corl,mask=np.zeros_like(corl, dtype=np.bool),cmap=sns.diverging_palette(220, 10, as_cmap=True),robust=True,square=True)  

##########################################################################################
############ Optimizing number of clusters for Agglomerative Clustering ##################

# Silhouette (optimun = 2)
plt.figure('optimun_n_of_clusters_Silhouette_Agglomerative_Clustering')
n_clusters = np.arange(2, 21)
models = [AgglomerativeClustering(n, affinity='manhattan', linkage='average').fit(Hierdf) for n in n_clusters]
plt.plot(n_clusters, [metrics.silhouette_score(Hierdf, m.labels_) for m in models],'r*-')
plt.suptitle('Optimizing of Agglomerative Clustering - Silhouette')
plt.xticks(np.arange(2, 21,2))
plt.xlabel('number of clusters')
plt.ylabel('Silhoiuette')

##########################################################################################
print("From the above looks like optimal number of clustering is 2")
   
"""

#Dendrogram
plt.figure('Dendrogram',figsize=(10, 7))  
plt.title("Dendrograms")  
dend = shc.dendrogram(shc.linkage(Hierdf, method='ward'))
plt.axhline(y=5.5, color='chocolate', linestyle='--')
plt.axhline(y=4.3, color='red', linestyle='--')

### Optimal model (n=3) ###
print("Optimal model:")
print("-----------------")
#Train the model with optimal parameters
ag = AgglomerativeClustering(n_clusters=3, affinity='manhattan', linkage='average')  
ag.fit(Hierdf)
#Predict the datum's classes
Ags1=ag.labels_
#Evaluate the model
silhouette_values = metrics.silhouette_score(Hierdf, Ags1)
print ("Silhouette: %0.3f" % silhouette_values)
n_clusters_AGC = len(set(Ags1)) 
print("\nEstimated number of clusters: %d" % n_clusters_AGC)

### Comparison model (n=2) ###
print("\nComparison model:")
print("-----------------")
#Train the model with comparison model
ag = AgglomerativeClustering(n_clusters=2, affinity='manhattan', linkage='average')  
ag.fit(Hierdf)
#Predict the datum's classes
Ags2=ag.labels_
#Evaluate the model
silhouette_values = metrics.silhouette_score(Hierdf, Ags2)
print ("Silhouette: %0.3f" % silhouette_values)
print("NMI for downgrading from 3 to 2 cluster is: %0.3f" %normalized_mutual_info_score(Ags1, Ags2))
n_clusters_cAGC = len(set(Ags2)) 
print("\nEstimated number of clusters: %d" % n_clusters_cAGC)

#Adding the columns to the Dataset
Hierdf["Cluster_Agglomerative"] = pd.DataFrame(Ags1)
Hierdf["Cluster(n=2)_Agglomerative"] = pd.DataFrame(Ags2)  

##########################################
##### Agglomerative Clustering plots #####
##########################################

### Calculate the cendroids for ploting

#Find Centroids
Agc = NearestCentroid()

#Optimal Model
AGO_centr=Agc.fit(Hierdf, Ags1).centroids_
#Comparison Model
AGC_centr=Agc.fit(Hierdf, Ags2).centroids_

### Optimal model (n=3) ### 
#Scatterplot
plt.figure('Scatterplot Agglomerative Optimal')
ax=sns.scatterplot(x=Hierdf['principal component 1'], y=Hierdf['principal component 2'],hue=Hierdf["Cluster_Agglomerative"],legend=False)
ax.scatter(x=Hierdf['principal component 1'], y=Hierdf['principal component 2'],c=Hierdf["Cluster_Agglomerative"],cmap='rainbow',alpha=0.9,)
plt.scatter(AGO_centr[:, 0], AGO_centr[:, 1], c='black', s=200, alpha=0.8)
ax.set_title("Agglomerative Clustering with 3 Components")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
#boxplot
plt.figure('Boxplot Agglomerative Optimal')
sns.boxplot(x='Cluster_Agglomerative', y='principal component 1',notch=True, data=Hierdf).set_title("Agglomerative Clustering Optimal Boxplot")

### Comparison model (n=2) ###
#Scatterplot
plt.figure('Scatterplot Agglomerative for Comparison')
ax=sns.scatterplot(x=Hierdf['principal component 1'], y=Hierdf['principal component 2'],hue=Hierdf["Cluster(n=2)_Agglomerative"],legend=False)
ax.scatter(Hierdf['principal component 1'], y=Hierdf['principal component 2'],c=Hierdf["Cluster(n=2)_Agglomerative"],cmap='rainbow',alpha=0.9,)
plt.scatter(AGC_centr[:, 0], AGC_centr[:, 1], c='black', s=200, alpha=0.8)
ax.set_title("Agglomerative Clustering  with 2 Components for Comparison")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
#boxplot
plt.figure('Boxplot Agglomerative for Comparison')
sns.boxplot(x='Cluster(n=2)_Agglomerative', y='principal component 1',notch=True, data=Hierdf).set_title("Agglomerative Clustering Boxplot for Comparison")

###### 3D plot #######

fig = plt.figure('3D Agglomerative',figsize=(20,20))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(Hierdf['principal component 1'][Hierdf.Cluster_Agglomerative == 0], Hierdf['principal component 2'][Hierdf.Cluster_Agglomerative == 0], Hierdf['principal component 3'][Hierdf.Cluster_Agglomerative == 0], c='blue',alpha=0.8, s=70)
ax.scatter(Hierdf['principal component 1'][Hierdf.Cluster_Agglomerative == 1], Hierdf['principal component 2'][Hierdf.Cluster_Agglomerative == 1], Hierdf['principal component 3'][Hierdf.Cluster_Agglomerative == 1], c='red',alpha=1, s=70)
ax.scatter(Hierdf['principal component 1'][Hierdf.Cluster_Agglomerative == 2], Hierdf['principal component 2'][Hierdf.Cluster_Agglomerative == 2], Hierdf['principal component 3'][Hierdf.Cluster_Agglomerative == 2], c='lime',alpha=0.8, s=70)
#ax.scatter(Hierdf['principal component 1'][Hierdf.Cluster_Agglomerative == 3], Hierdf['principal component 2'][Hierdf.Cluster_Agglomerative == 3], Hierdf['principal component 3'][Hierdf.Cluster_Agglomerative == 3], c='red',alpha=1, s=70)
#ax.scatter(Hierdf['principal component 1'][Hierdf.Cluster_Agglomerative == 4], Hierdf['principal component 2'][Hierdf.Cluster_Agglomerative == 4], Hierdf['principal component 3'][Hierdf.Cluster_Agglomerative == 4], c='black',alpha=1, s=70)
ax.view_init(30, 185)
ax.set_title("Agglomerative Clustering 3D")
ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
ax.set_zlabel("Principal Component 3")
plt.show()
 
########################################################
######### Fitting model in Raw dataset #################
########################################################
# not perfomed very good
d["Cluster_Gaussian"]=pd.DataFrame(gm_y1)
d.fillna(6,inplace=True)
fig = plt.figure('3D test in raw dataset',figsize=(20,20))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(d['Channel'][d.Cluster_Gaussian == 0],d['Detergents_Paper'][d.Cluster_Gaussian == 0], d['Grocery'][d.Cluster_Gaussian == 0],  c='blue',alpha=0.35, s=300)
ax.scatter(d['Channel'][d.Cluster_Gaussian == 1],d['Detergents_Paper'][d.Cluster_Gaussian == 1], d['Grocery'][d.Cluster_Gaussian == 1],  c='red',alpha=0.5, s=300)
ax.scatter(d['Channel'][d.Cluster_Gaussian == 2],d['Detergents_Paper'][d.Cluster_Gaussian == 2], d['Grocery'][d.Cluster_Gaussian == 2],  c='lime',alpha=0.5, s=300)
ax.scatter(d['Channel'][d.Cluster_Gaussian == 3],d['Detergents_Paper'][d.Cluster_Gaussian == 3], d['Grocery'][d.Cluster_Gaussian == 3],  c='yellow',alpha=0.9, s=300)
ax.scatter(d['Channel'][d.Cluster_Gaussian == 4],d['Detergents_Paper'][d.Cluster_Gaussian == 4], d['Grocery'][d.Cluster_Gaussian == 4],  c='cyan',alpha=0.35, s=300)
ax.scatter(d['Channel'][d.Cluster_Gaussian == 5],d['Detergents_Paper'][d.Cluster_Gaussian == 5], d['Grocery'][d.Cluster_Gaussian == 5],  c='green',alpha=0.5, s=350)
ax.scatter(d['Channel'][d.Cluster_Gaussian == 6],d['Detergents_Paper'][d.Cluster_Gaussian == 6], d['Grocery'][d.Cluster_Gaussian == 6],  c='black',alpha=1, s=40)
ax.view_init(30, 185)
ax.set_title("3D test in raw dataset")
ax.set_xlabel("Channel")
ax.set_ylabel("Detergents_Paper")
ax.set_zlabel("Grocery")
plt.show()

print("____________________________________")
print("___________________________________________________________________")