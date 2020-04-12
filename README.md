PART 1: Arrythmia Classification
================================================================================
As the problem is split into two parts, in the project folder reside 2 python scripts
(part_a_boolean.py,part_a_classes.py).

Running part_a_boolean.py performs classification on the boolean problem regarding the Arrythmia existence.

Executing part_a_classes.py explores the patients with a type of Arrythmia and 
performs classification the Arrythmia class-type.

Arrythmia.data is the file where our Arrythmia dataset resides.
Arrythmia.txt provides a data description on the Arrythmia dataset.

PART 2: Unsupervised Learning - Clustering 
================================================================================
For the second part of this project, we tackle an unsupervised problem concerning a customer segmentation problem. 
The dataset is from http://archive.ics.uci.edu/ml/datasets/Wholesale+customers.
 
Due to the length of the script, as a first step please run the command in line 36. 
The you can execute the hole code. 

A small index of the code:
1)	Initial we import the libraries and the dataset. (lines 1- 43)
2)	Plot the dataset and calculate the statistics. (lines 44- 127)
3)	Preprocessing the data and plot the same graphs for comparison (lines 128-224)
4)	 Clustering algorithms where (lines 225-798)
a.	DBSCAN (lines 233-336)
i.	Optimizing the Hyperparameters (lines 241-267)
ii.	Train the algorithm (lines 268-299)
iii.	Plot the results (lines 300-336)
b.	KMEANS (lines 336-488)
i.	Optimizing the Hyperparameters (lines 345-387)
ii.	Train the algorithm (optimal & comparison models) (lines 388-430)
iii.	Plot the results (lines 431-488)
c.	Gaussian Mixtures (lines 489-643)
i.	Optimizing the Hyperparameters (lines 489-536)
ii.	Train the algorithm (optimal & comparison models) (lines 537-582)
iii.	Plot the results (lines 583-643)
d.	Agglomerative Clustering (lines 644-774)
i.	Optimizing the Hyperparameters (lines 653-677)
ii.	Train the algorithm (optimal & comparison models) (lines 679-718)
iii.	Plot the results (lines 719-774)
5)	Fitting best model to the raw dataset (lines 775-798)


PART 3: Regression Affairs
================================================================================
- Datasource: https://fairmodel.econ.yale.edu/vote2012/affairs.txt
- regression_affairs.py contains code to generate most plots
and data for affairs_PT (or simply PT)
- processRB.py is code to generate the affairs_RB dataset
from a copied section of the datasource link
- plot_test.py includes the code that is responsible for 
drawing double bar plot figures found in the report
- affairs_util.py: utility functions
- For the ternary problem use the affairs_three_classes.py
- affairs_RB.py is responsible for Redbook dataset (RB)
- affairs_PT contains code that generates the PCA explained variance
graphs
- affairs_pca.py: on PT data, perform PCA using 4 components
- affairs_oversampling.py: oversampling using reshape() / best R2
value with polynomial degree=3
- affairs_log_binary: Solution to the binary classification problem


PART 4: Scaling up - Predicting Buys
================================================================================
As the problem is split into two parts, in the project folder reside 2 python scripts
(Dataset_creation.py,Classification.py).

In order to test the problem first we run the Dataset_creation.py to export a new dataset 
in a csv file and then we execute Classification.py script to import the new dataset and 
run classification on it.

Balanced_class_Dataset(80-20).csv is an already created Dataset ready for classification testing

-Original Dataset


This dataset was constructed by YOOCHOOSE GmbH to support participants in the RecSys Challenge 2015.
See  http://recsys.yoochoose.net for details about the challenge.

The YOOCHOOSE dataset contain a collection of sessions from a retailer, where each session
is encapsulating the click events that the user performed in the session.
For some of the sessions, there are also buy events; means that the session ended
with the user bought something from the web shop. The data was collected during several
months in the year of 2014, reflecting the clicks and purchases performed by the users
of an on-line retailer in Europe.  To protect end users privacy, as well as the retailer,
all numbers have been modified. Do not try to reveal the identity of the retailer.

-LICENSE

This dataset is licensed under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0
International License. To view a copy of this license, visit http://creativecommons.org/licenses/by-nc-nd/4.0/.
YOOCHOOSE cannot guarantee the completeness and correctness of the data or the validity
of results based on the use of the dataset as it was collected by implicit tracking of a website. 
If you have any further questions or comments, please contact YooChoose <support@YooChoose.com>. 
The data is provided "as it is" and there is no obligation of YOOCHOOSE to correct it,
improve it or to provide additional information about it.

-CLICKS DATASET FILE DESCRIPTION

The file yoochoose-clicks.dat comprising the clicks of the users over the items.
Each record/line in the file has the following fields/format: Session ID, Timestamp, Item ID, Category
-Session ID – the id of the session. In one session there are one or many clicks. Could be represented as an integer number.
-Timestamp – the time when the click occurred. Format of YYYY-MM-DDThh:mm:ss.SSSZ
-Item ID – the unique identifier of the item that has been clicked. Could be represented as an integer number.
-Category – the context of the click. The value "S" indicates a special offer, "0" indicates  a missing value, a number between 1 to 12 indicates a real category identifier,
 any other number indicates a brand. E.g. if an item has been clicked in the context of a promotion or special offer then the value will be "S", if the context was a brand i.e BOSCH,
 then the value will be an 8-10 digits number. If the item has been clicked under regular category, i.e. sport, then the value will be a number between 1 to 12. 
 
-BUYS DATSET FILE DESCRIPTION
=
The file yoochoose-buys.dat comprising the buy events of the users over the items.
Each record/line in the file has the following fields: Session ID, Timestamp, Item ID, Price, Quantity

-Session ID - the id of the session. In one session there are one or many buying events. Could be represented as an integer number.
-Timestamp - the time when the buy occurred. Format of YYYY-MM-DDThh:mm:ss.SSSZ
-Item ID – the unique identifier of item that has been bought. Could be represented as an integer number.
-Price – the price of the item. Could be represented as an integer number.
-Quantity – the quantity in this buying.  Could be represented as an integer number.


 
