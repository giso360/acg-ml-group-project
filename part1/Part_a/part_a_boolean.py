
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 17:50:11 2020

@author: Dimitrios  Papaioannou
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, roc_auc_score


#Reading the dataset from the file and defining the class column
data=pd.read_csv('arrhythmia.data',header=None)
Class_column=data.loc[:,data.shape[1]-1]

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
#data=sel.fit_transform(data)


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

Class_column=data[:,data.shape[1]-1]

#Transforming our classification problem into boolean (Existence of arrythmia or not)
Boolean_Class=np.where(Class_column!=1,0,Class_column)

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

pca1 = PCA(n_components=40,svd_solver='full',random_state=1)
features=pca1.fit(features).transform(features)

#Split of the dataset into training set  and testing set
X_train, X_test, Y_train, Y_test = train_test_split (features,Boolean_Class, test_size=0.2, random_state=1)

#-----------------------------#
#        Classification   	  #
#-----------------------------#

#Define Decision Tree
Depths_Leaves=[(10,20),(10,5),(4,10),(20,60)]
fposDT,trposDT,threshDT=[],[],[]
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
    fprDT, tprDT, thresholdsDT = roc_curve(Y_test, pr_y_test_pred_DT[:,1])
    fposDT.append(fprDT)
    trposDT.append(tprDT)
    threshDT.append(thresholdsDT)
    
##Export a more simple/readable tree graph which has similar performace regarding Precision/Recall/F1
clfDT =  tree.DecisionTreeClassifier(max_depth=4,max_leaf_nodes=10,random_state=1)
clfDT.fit(X_train, Y_train)
dot_data = tree.export_graphviz(clfDT, out_file=None) 
tree.export_graphviz(clfDT, out_file='WBC-tree1.dot')

#ROC curve for different Decision Trees
lwidth=2
plt.figure(1)
plt.plot(fposDT[0],trposDT[0],color='blue',label='Decision Tree Depth:10,Leaves:20')
plt.plot(fposDT[1],trposDT[1],color='green',label='Decision Tree Depth:10,Leaves:5')
plt.plot(fposDT[2],trposDT[2],color='red',label='Decision Tree Depth:4,Leaves:10')
plt.plot(fposDT[3],trposDT[3],color='black',label='Decision Tree Depth:13,Leaves:53')
plt.plot([0, 1], [0, 1], color='navy', lw=lwidth, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve on different Decision Trees')
plt.legend(loc="lower right")
plt.show()


#Define Neural Network
Layers=[(100,),(5,2,),(3,3,2,2),(50,50)]
fposANN,trposANN,threshANN=[],[],[]
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
    fprANN, tprANN, thresholdsANN = roc_curve(Y_test, pr_y_test_pred_ANN[:,1])
    fposANN.append(fprANN)
    trposANN.append(tprANN)
    threshANN.append(thresholdsANN)
    
#ROC curve for different Neural Networks
lwidth=2
plt.figure(2)
plt.plot(fposANN[0],trposANN[0],color='blue',label='Neural Network:'+str(Layers[0]))
plt.plot(fposANN[1],trposANN[1],color='green',label='Neural Network:'+str(Layers[1]))
plt.plot(fposANN[2],trposANN[2],color='red',label='Neural Network:'+str(Layers[2]))
plt.plot(fposANN[3],trposANN[3],color='black',label='Neural Network:'+str(Layers[3]))
plt.plot([0, 1], [0, 1], color='navy', lw=lwidth, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve on different Neural Networks')
plt.legend(loc="lower right")
plt.show()



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
    fprRF, tprRF, thresholdsRF = roc_curve(Y_test, pr_y_test_pred_RF[:,1])
    fposRF.append(fprRF)
    trposRF.append(tprRF)
    threshRF.append(thresholdsRF)
    
#ROC curve for different Random Forest scenarios
lwidth=2
plt.figure(2)
plt.plot(fposRF[0],trposRF[0],color='blue',label='Random Forest-Trees#:'+str(Number_of_Trees[0]))
plt.plot(fposRF[1],trposRF[1],color='green',label='Random Forest-Trees#:'+str(Number_of_Trees[1]))
plt.plot(fposRF[2],trposRF[2],color='red',label='Random Forest-Trees#:'+str(Number_of_Trees[2]))
plt.plot(fposRF[3],trposRF[3],color='black',label='Random Forest-Trees#:'+str(Number_of_Trees[3]))
plt.plot([0, 1], [0, 1], color='navy', lw=lwidth, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve on different Neural Networks')
plt.legend(loc="lower right")
plt.show()

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


pr_y_test_pred_DT = clfDT.predict_proba(X_test)
pr_y_test_pred_ANN = clfANN.predict_proba(X_test)
pr_y_test_pred_NB = clfNB.predict_proba(X_test)
pr_y_test_pred_LinSVM = clflinearSVM.predict_proba(X_test)
pr_y_test_pred_SVM = clfSVM.predict_proba(X_test)
pr_y_test_pred_RndF = clfRF.predict_proba(X_test)


#ROC CURVE ON OUR BEST CLASSIFICATION SCENARIOS

fprDT, tprDT, thresholdsDT = roc_curve(Y_test, pr_y_test_pred_DT[:,1])
fprANN, tprANN, thresholdsANN = roc_curve(Y_test, pr_y_test_pred_ANN[:,1])
fprNB, tprNB, thresholdsNB = roc_curve(Y_test, pr_y_test_pred_NB[:,1])
fprLinSVM, tprLinSVM, thresholdsLinSVM = roc_curve(Y_test, pr_y_test_pred_LinSVM[:,1])
fprSVM, tprSVM, thresholdsSVM = roc_curve(Y_test, pr_y_test_pred_SVM[:,1])
fprRndf, tprRndf, thresholdsRndf = roc_curve(Y_test, pr_y_test_pred_RndF[:,1])



lwidth=2
plt.figure(3)
plt.plot(fprDT,tprDT,color='blue',label='Decision Tree')
plt.plot(fprANN,tprANN,color='green',label='Neural Network')
plt.plot(fprNB,tprNB,color='red',label='Naive Bayes')
plt.plot(fprLinSVM,tprLinSVM,color='black',label='Linear SVM')
plt.plot(fprSVM,tprSVM,color='yellow',label='SVM')
plt.plot(fprRndf,tprRndf,color='orange',label='Random Forest')
plt.plot([0, 1], [0, 1], color='navy', lw=lwidth, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve on our Classifiers')
plt.legend(loc="lower right")
plt.show()



#ROC CURVE LINEAR SVM VS GAUSSIAN SVM VS NAIVE BAYES
lwidth=2
plt.figure(3)
plt.plot(fprNB,tprNB,color='red',label='Naive Bayes')
plt.plot(fprLinSVM,tprLinSVM,color='black',label='Linear SVM')
plt.plot(fprSVM,tprSVM,color='yellow',label='SVM')
plt.plot([0, 1], [0, 1], color='navy', lw=lwidth, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve on our Classifiers')
plt.legend(loc="lower right")
plt.show()



#Implementing  AUC SCORES for the best classification scenarios
print("AUC SCORES")
print('Decision Tree : ',roc_auc_score(Y_test,y_test_pred_DT))
print('Naive Bayes : ',roc_auc_score(Y_test,y_test_pred_NB))
print('Neural Network : ',roc_auc_score(Y_test,y_test_pred_ANN))
print('Linear SVM : ',roc_auc_score(Y_test,y_test_pred_LinSVM))
print('SVM : ',roc_auc_score(Y_test,y_test_pred_SVM))
print('Random Forest : ',roc_auc_score(Y_test,y_test_pred_RF))

