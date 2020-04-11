import pandas as pd
import numpy as np

#Function to create day time categorical value
def my_day_time(day_time):
  if day_time>=6 and day_time<12:
      return 'Morning'
  elif day_time>=12 and day_time<18:
      return 'Afternoon'
  elif day_time>=18:
      return 'Evening'
  else:
      return 'Night'

#Function to determine the Month term
def my_month(month):
    if month==4 or month==5 or month==6:
        return 'Apr/May/June'
    else:
        return 'July/Aug/Sept'
    
#Function to observe if the session occurs during weekend or not
def my_week_days(day):
    if day<5:
        return 0
    else:
        return 1
    
#Function to denote the 
def S_occur(group):
    cnt=0
    for item in group:
        if item == 'S':
            cnt+=1
    return cnt
#Function to calculate how many items of the Session are popular
def Pop_items_perc(group,popularity):
    cnt=0
    for item in group:
        if item in popularity:
            cnt+=1
    return cnt


#Reading the datasets from the two files
data_buys = pd.read_csv('yoochoose-buys.dat',header=None)
data_clicks = pd.read_csv('yoochoose-clicks.dat',dtype={3:object},header=None)

#Transforming timestamp feature to type datetime64
data_clicks[1] = pd.to_datetime(data_clicks[1])

#Retaining the day,the hour of the day and  the month the Session took place     
daytime= data_clicks.groupby(0).first()[1].dt.hour
Months= data_clicks.groupby(0).first()[1].dt.month
Week_days= data_clicks.groupby(0).first()[1].dt.weekday


#Applying one hot encoding on categorical values "daytime,Months,Week_days"
Daytime_buckles= pd.get_dummies(daytime.apply(my_day_time).values).astype("int")
Month_buckles = pd.get_dummies(Months.apply(my_month).values).astype("int")
Weekend_days = pd.Series(Week_days.apply(my_week_days).values)


#
itemIDS = np.in1d(data_clicks[2].unique(),data_buys[2].unique()).astype('int')

#Unique click Sessions ID for each we thrive to discover the buy_outcome
Sessions = pd.Series(data_clicks[0].unique()).sort_values().values

#Number of clicks of each session
Clicks_number = data_clicks[0].value_counts(sort=False).values

#How many unique items  the session has 
Distinct_items = data_clicks.groupby(0)[2].nunique().values

#Duration of each session in minutes
Session_duration=data_clicks.groupby(0)[1].apply(lambda x: x.max()-x.min())/np.timedelta64(1,'m')

#Creating a popularity index out of all items and obtaining only the top 10%
Item_Appearances = data_clicks[2].value_counts()

Popular_items = Item_Appearances[0:int(len(Item_Appearances)/10)]


#Calculating the Item popularity percentage of each session
Sess_Items_No = data_clicks.groupby(0)[2].count()
Sess_Popular_items = data_clicks.groupby(0)[2].apply(lambda x: Pop_items_perc(x,Popular_items))
Sess_Items_Popularity = (Sess_Popular_items/Sess_Items_No).values


#Calculating the percentage of 'S' ( Special offer) items in the session
Categories = data_clicks.groupby(0)[3].count()
S_category = data_clicks.groupby(0)[3].apply(S_occur)
S_perc = (S_category/Categories).values

# Buy outcome denoted by 0 (No buy) or 1 (Buy) derived from the unique Session IDs that reside in both clicks and buys
Class_column = np.in1d(Sessions,data_buys[0].unique()).astype('int')

#Creating the dataframe and assignment of the features we created.
Data=pd.DataFrame()

Data['Session_ID'] = Sessions
Data['Clicks_#'] = Clicks_number
Data['Duration(min)'] = Session_duration.values
Data['Distinct_items'] = Distinct_items
Data['Spec_Offers']= S_perc
Data['Items_popularity']= Sess_Items_Popularity 
Data['Weekend'] = Weekend_days

Data=pd.concat([Data,Daytime_buckles],axis=1)
Data=pd.concat([Data,Month_buckles],axis=1)

Data['Buy_Outcome']=Class_column


#Rebalancing the dataset using less records for the dominant class label (Undersampling the 0 label)
buys = len(Data[Data['Buy_Outcome'] == 1])
buys_indices = Data[Data.Buy_Outcome == 1].index

#How many more sets of the dominant class we use versus the low sample class n=4 --> 80%/20% class balance
num=4

Non_buys_indices= Data[Data.Buy_Outcome == 0].index

random_indices = np.random.choice(Non_buys_indices,buys*num, replace=False)

under_sample_indices = np.concatenate([buys_indices,random_indices])

Balanced_class_dataset = Data.loc[under_sample_indices]

#Shuffling and exporting the dataset
Balanced_class_dataset=Balanced_class_dataset.sample(frac=1)

Balanced_class_dataset.to_csv(r'Balanced_class_Dataset(80-20).csv', index = False)
