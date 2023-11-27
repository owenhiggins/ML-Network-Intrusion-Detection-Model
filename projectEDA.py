import pandas as pd
from collections import Counter


#import the base dataset
data = pd.read_csv(r'C:/project/data/NF-UNSW-NB15.csv')

#view first rows, check for read errors
print(data.head())

#look at shape of data
print(data.shape)

#more info on dataset
print(data.info())


#more detailed info on dataset
print(data.describe().to_string())


#check for nulls
print(data.isnull().sum().sort_values(ascending = False))


#check for duplicates, load into list
dupes = data.duplicated().tolist()

#run list through Counter object, print unique values
#should be True or False for dupes
print(Counter(dupes).keys())


#print counts of Trues and Falses
print(Counter(dupes).values())







