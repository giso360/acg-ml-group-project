# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 00:08:01 2020

@author: Dimitrios Papaioannou
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import tree

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_curve, roc_auc_score

#Loading the balanced set created in the Dataset_creation.py file
Balanced_class_dataset=pd.read_csv('Balanced_class_Dataset(80-20).csv', header=0)

#Not really wanting Id column in the training set
Balanced_class_dataset.drop('Session_ID',axis=1,inplace=True)



#Split of the dataset into training set  and testing set
X_train, X_test, Y_train, Y_test = train_test_split ( Balanced_class_dataset.iloc[:,:-1],Balanced_class_dataset['Buy_Outcome'], test_size=0.3, random_state=1)


#Define Decision Tree
Depths_Leaves=[(10,5),(3,7),(30,1000),(10,20)]
fposDT,trposDT,threshDT=[],[],[]
for item in Depths_Leaves:
    print(item)
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
    print ('Macro Precision: ',precision_recall_fscore_support(Y_test, y_test_pred_DT, average='macro')[0])
    print ('Macro Recall: ',precision_recall_fscore_support(Y_test, y_test_pred_DT, average='macro')[1])
    print ('Macro F1_Score: ',precision_recall_fscore_support(Y_test, y_test_pred_DT, average='macro')[2],'\n')
    fprDT, tprDT, thresholdsDT = roc_curve(Y_test, pr_y_test_pred_DT[:,1])
    fposDT.append(fprDT)
    trposDT.append(tprDT)
    threshDT.append(thresholdsDT)
    

##Export a more simple/readable tree graph which has similar performace regarding Precision/Recall/F1
clfDT =  tree.DecisionTreeClassifier(max_depth=3,max_leaf_nodes=7,random_state=1)
clfDT.fit(X_train, Y_train)
dot_data = tree.export_graphviz(clfDT, out_file=None) 
tree.export_graphviz(clfDT, out_file='WBC-tree.dot')



#ROC curve for different Decision Trees
lwidth=2
plt.figure(1)
plt.plot(fposDT[0],trposDT[0],color='blue',label='Decision Tree Depth:10,Leaves:20')
plt.plot(fposDT[1],trposDT[1],color='green',label='Decision Tree Depth:10,Leaves:5')
plt.plot(fposDT[2],trposDT[2],color='red',label='Decision Tree Depth:4,Leaves:10')
plt.plot(fposDT[3],trposDT[3],color='black',label='Decision Tree Depth:'+str(clfDT.get_depth())+',Leaves:'+str(clfDT.get_n_leaves()))
plt.plot([0, 1], [0, 1], color='navy', lw=lwidth, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve on different Decision Trees')
plt.legend(loc="lower right")
plt.show()



#Define Neural Network
Layers=[(7,5),(5,4),(5,5,)]
fposANN,trposANN,threshANN=[],[],[]
for item in Layers:
    clfANN = MLPClassifier(solver='lbfgs', activation='relu',
                    batch_size=1, tol=1e-03,
                      hidden_layer_sizes=item, random_state=1, max_iter=1000000, verbose=True)
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
plt.plot([0, 1], [0, 1], color='navy', lw=lwidth, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve on different Neural Networks')
plt.legend(loc="lower right")
plt.show()


#Define a Naive Bayes
clfNB = GaussianNB()
#Training our remaining Classifiers                     
clfNB.fit(X_train,Y_train)

y_test_pred_NB = clfNB.predict(X_test)

#Confusion matrix of our models towards the test data
confMatrix_Test_NB = confusion_matrix(Y_test, y_test_pred_NB, labels=None)

print ('Naive Bayes Classifier')
print ('Confusion Matrix')
print (confMatrix_Test_NB,'\n')

print ('Macro Precision:%.3f'%precision_recall_fscore_support(Y_test, y_test_pred_NB, average='macro')[0])
print ('Macro Recall:%.3f '%precision_recall_fscore_support(Y_test, y_test_pred_NB, average='macro')[1])
print ('Macro F1_Score:%.3f '%precision_recall_fscore_support(Y_test, y_test_pred_NB, average='macro')[2],'\n')


pr_y_test_pred_NB = clfNB.predict_proba(X_test)


#ROC CURVE ON OUR BEST CLASSIFICATION SCENARIOS

fprNB, tprNB, thresholdsNB = roc_curve(Y_test, pr_y_test_pred_NB[:,1])

lwidth=2
plt.figure(3)
plt.plot(fprDT,tprDT,color='blue',label='Decision Tree')
plt.plot(fprNB,tprNB,color='red',label='Naive Bayes')
plt.plot(fprANN,tprANN,color='green',label='Neural Networks')
plt.plot([0, 1], [0, 1], color='navy', lw=lwidth, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve on our Classifiers')
plt.legend(loc="lower right")
plt.show()

#IMPLEMENTING AUC SCORES for the best classification scenarios
print("AUC SCORES")
print('Decision Tree : ',roc_auc_score(Y_test,y_test_pred_DT))
print('Naive Bayes : ',roc_auc_score(Y_test,y_test_pred_NB))
print('Neural Network : ',roc_auc_score(Y_test,y_test_pred_ANN))

