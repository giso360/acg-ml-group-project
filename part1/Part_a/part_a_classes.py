# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 21:23:32 2020

@author: Papahs
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn import svm

from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn import tree

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

from collections import Counter
from imblearn.over_sampling import SMOTE 

#Reading the dataset from the file
data=pd.read_csv('arrhythmia.data',header=None)

#Having solved the boolean problem of arrhythmia existence we tend to classify the types of arrythmias
#So we remove records with no arrhythmia
data = data.loc[data[279] != 1]

Class_column=data.loc[:,data.shape[1]-1]

#merging  Old Anterior/Old Inferior Myocardial Infarction  arrythmia classes 
Class_column.replace(3,4,inplace=True)

#merging  Left/Right bundle branch block arrythmia classes 
Class_column.replace(9,10,inplace=True)

#merging   Sinus tachycardy/bradycardy arrhythmia classes
Class_column.replace(5,6,inplace=True)

#Merging really low samplewise classes with others
b=Class_column.value_counts()
Low_samples=list(b[b<10].index)
Class_column.replace(Low_samples,16,inplace=True)


#Data description
print ('Data report:')
print('\n#Classes=',len(set(Class_column)))
a=Class_column.groupby(Class_column).count()

for Class,samples in a.items():
    print(f'Class-{Class} #number of samples={samples}')
    
print ('\n#Data samples=',data.shape[0])
print ('#Data attributes=',data.shape[1]-1)   
print('------------------------------------------')


#-----------------------------#
#        PreProcessing   	  #
#-----------------------------#

#Replacing the missing values with a numpy nan
data.replace('?',np.NaN,inplace=True)

#Getting the percentage of missing values per column
Missing=pd.DataFrame({'percent_missing': data.isnull().sum() * 100 / len(data)})

#Two ways to drop the columns with a percentage of missing values higher than 70%
data.drop(data.columns[data.apply(lambda x: x.isnull().sum()*100/len(data)>70)],axis=1,inplace=True)
#data = data.loc[:, (data.isnull().sum()<60)]

#Replacing the numpy Nan values with themean of each feature/column
data = data.astype('float64')
data.fillna(data.mean(),inplace=True)

#Observing the data we drop boolean features(0/1) that are dominated by the zero value.
data=data.loc[:,((data==0).sum()/len(data)<0.94)]

#---------------------------------#
#        Feature Selection   	  #
#---------------------------------#

#CORRELATION
Corr_matrix = data.corr().abs()
s = Corr_matrix .unstack().sort_values(ascending=False).drop_duplicates()
s = s[s.apply(lambda x: x>0.9 and x<1)]

# display the highly correlated features
High_Corr=list(s.index.get_level_values(0))
data.drop(High_Corr,axis=1,inplace=True)

data=np.array(data)

#SCALING
#Applying the scale function to standarize the features of the dataset (normalization)
features=preprocessing.scale(data[:,0:data.shape[1]-1])

#PCA
pca = PCA(svd_solver='full',random_state=1)
features=pca.fit(features).transform(features)
a=np.cumsum(pca.explained_variance_ratio_)
#Plotting the Cumulative Summation of the Explained Variance regarding the dataset
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)')
plt.title('Dataset PCA Explained Variance')
plt.show()

pca1 = PCA(n_components=30,svd_solver='full',random_state=1)
features=pca1.fit(features).transform(features)

#Split of the dataset into training set  and testing set
x_train, X_test, y_train, Y_test = train_test_split (features,Class_column, test_size=0.5, random_state=1)


#Implementing SMOTE for over sampling low sample class records

print('Original dataset shape %s' % Counter(y_train))
sm = SMOTE(random_state=1)
X_train, Y_train = sm.fit_resample(x_train,y_train)
print('Resampled dataset shape after enforcing SMOTE%s\n' % Counter(Y_train))

#-----------------------------#
#        Classification   	  #
#-----------------------------#

#Define Decision Tree
Depths_Leaves=[(10,20),(10,5),(4,10),(20,60)]
for item in Depths_Leaves: 
    clfDT =  tree.DecisionTreeClassifier(max_depth=item[0],max_leaf_nodes=item[1],random_state=1)
    #Training the classifiers
    clfDT.fit(X_train, Y_train)
    #Test the trained model on the test set
    y_test_pred_DT=clfDT.predict(X_test)
    #Confusion matrix of our model towards the test data
    confMatrix_Test_DT = confusion_matrix(Y_test, y_test_pred_DT, labels=None)
    
    print (f'Decision Tree Depth: {clfDT.get_depth()}, Leaves: {clfDT.get_n_leaves()}')
    print ('Confusion Matrix')
    print (confMatrix_Test_DT,'\n')
    
    pr_y_test_pred_DT=clfDT.predict_proba(X_test)
    #Measures of performance: Precision, Recall, F1
    print ('Macro Precision:%.3f '%precision_recall_fscore_support(Y_test, y_test_pred_DT, average='macro')[0])
    print ('Macro Recall:%.3f '%precision_recall_fscore_support(Y_test, y_test_pred_DT, average='macro')[1])
    print ('Macro F1_Score:%.3f '%precision_recall_fscore_support(Y_test, y_test_pred_DT, average='macro')[2],'\n')

#Define Neural Network
Layers=[(100,),(5,2,),(3,3,2,2),(50,50)]
for item in Layers:
    clfANN = MLPClassifier(solver='lbfgs', activation='relu',
                    batch_size=1, tol=1e-05,
                     hidden_layer_sizes=item, random_state=1, max_iter=10000, verbose=True)
    clfANN.fit(X_train, Y_train)
   
    y_test_pred_ANN=clfANN.predict(X_test)
    
    confMatrix_Test_ANN = confusion_matrix(Y_test, y_test_pred_ANN, labels=None)
    
    print (f'Neural network {item}')
    print ('Confusion Matrix')
    print (confMatrix_Test_ANN,'\n')
    
    pr_y_test_pred_ANN=clfANN.predict_proba(X_test)
    
    print ('Macro Precision:%.3f'%precision_recall_fscore_support(Y_test, y_test_pred_ANN, average='macro')[0])
    print ('Macro Recall:%.3f '%precision_recall_fscore_support(Y_test, y_test_pred_ANN, average='macro')[1])
    print ('Macro F1_Score:%.3f '%precision_recall_fscore_support(Y_test, y_test_pred_ANN, average='macro')[2],'\n')
    
#Random Forest Classifier
Number_of_Trees=[3,20,50,100]
fposRF,trposRF,threshRF=[],[],[]
for item in Number_of_Trees:
    clfRF=RandomForestClassifier(n_estimators=item)
    clfRF.fit(X_train, Y_train)
    
    y_test_pred_RF=clfRF.predict(X_test)
    
    confMatrix_Test_RF = confusion_matrix(Y_test, y_test_pred_RF, labels=None)
    
    print (f'Random Forest_{item} estimators')
    print ('Confusion Matrix')
    print (confMatrix_Test_RF,'\n')
    
    pr_y_test_pred_RF=clfRF.predict_proba(X_test)
    
    print ('Macro Precision:.%3f '%precision_recall_fscore_support(Y_test, y_test_pred_RF, average='macro')[0])
    print ('Macro Recall:%.3f '%precision_recall_fscore_support(Y_test, y_test_pred_RF, average='macro')[1])
    print ('Macro F1_Score:.%3f '%precision_recall_fscore_support(Y_test, y_test_pred_RF, average='macro')[2],'\n')



#Define Gaussian Support vector machine
clfSVM= svm.SVC(C=1, cache_size=200, class_weight=None, coef0=0.0,
decision_function_shape='ovo', degree=3, gamma='auto', kernel='rbf',
max_iter=-1, probability=True, random_state=None, shrinking=True,
tol=0.001, verbose=False)

#Define Linear Support Vector machine
clflinearSVM= svm.SVC(C=1, cache_size=200, class_weight=None, coef0=0.0,
decision_function_shape='ovo', degree=3, gamma='auto', kernel='linear',
max_iter=-1, probability=True, random_state=None, shrinking=True,
tol=0.001, verbose=False)

#Define a Naive Bayes
clfNB = GaussianNB()


#Training our Classifiers                     
clfNB.fit(X_train,Y_train)
clfSVM.fit(X_train, Y_train)
clflinearSVM.fit(X_train, Y_train)

y_test_pred_NB = clfNB.predict(X_test)
y_test_pred_SVM = clfSVM.predict(X_test)
y_test_pred_LinSVM = clflinearSVM.predict(X_test)


#Define Support vector machine
clfSVM= svm.SVC(C=10.0, cache_size=200, class_weight=None, coef0=0.0,
decision_function_shape='ovo', degree=3, gamma='auto', kernel='rbf',
max_iter=-1, probability=True, random_state=None, shrinking=True,
tol=0.001, verbose=False)

#Linear Support Vector machine
clflinearSVM= svm.SVC(C=10.0, cache_size=200, class_weight=None, coef0=0.0,
decision_function_shape='ovo', degree=3, gamma='auto', kernel='linear',
max_iter=-1, probability=True, random_state=None, shrinking=True,
tol=0.001, verbose=False)

#Define a Naive Bayes
clfNB = GaussianNB()

#Training our remaining Classifiers                     
clfNB.fit(X_train,Y_train)
clflinearSVM.fit(X_train, Y_train)
clfSVM.fit(X_train, Y_train)

y_test_pred_NB = clfNB.predict(X_test)
y_test_pred_SVM = clfSVM.predict(X_test)
y_test_pred_LinSVM = clflinearSVM.predict(X_test)


#Confusion matrix of our models towards the test data
confMatrix_Test_NB = confusion_matrix(Y_test, y_test_pred_NB, labels=None)
confMatrix_Test_SVM = confusion_matrix(Y_test, y_test_pred_SVM, labels=None)
confMatrix_Test_LinSVM = confusion_matrix(Y_test, y_test_pred_LinSVM, labels=None)

print ('Naive Bayes Classifier')
print ('Confusion Matrix')
print (confMatrix_Test_NB,'\n')

print ('Macro Precision:%.3f'%precision_recall_fscore_support(Y_test, y_test_pred_NB, average='macro')[0])
print ('Macro Recall:%.3f '%precision_recall_fscore_support(Y_test, y_test_pred_NB, average='macro')[1])
print ('Macro F1_Score:%.3f '%precision_recall_fscore_support(Y_test, y_test_pred_NB, average='macro')[2],'\n')



print ('SVM Classifier')
print ('Confusion Matrix')
print (confMatrix_Test_SVM,'\n')

print ('Macro Precision:%.3f '%precision_recall_fscore_support(Y_test, y_test_pred_SVM, average='macro')[0])
print ('Macro Recall:%.3f '%precision_recall_fscore_support(Y_test, y_test_pred_SVM, average='macro')[1])
print ('Macro F1_Score:%.3f '%precision_recall_fscore_support(Y_test, y_test_pred_SVM, average='macro')[2],'\n')



print ('LinearSVM Classifier')
print ('Confusion Matrix')
print (confMatrix_Test_LinSVM,'\n')

print ('Macro Precision:%.3f '%precision_recall_fscore_support(Y_test, y_test_pred_LinSVM, average='macro')[0])
print ('Macro Recall:%.3f '%precision_recall_fscore_support(Y_test, y_test_pred_LinSVM, average='macro')[1])
print ('Macro F1_Score:%.3f '%precision_recall_fscore_support(Y_test, y_test_pred_LinSVM, average='macro')[2],'\n')
