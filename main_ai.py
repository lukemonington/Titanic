import pandas as pd
pd.set_option('max_columns', None)
import numpy as np


# Get train and test dataset
train = pd.read_csv(r'C:\Users\lukem\Desktop\Github AI Projects\Titanic\train.csv')
test = pd.read_csv(r'C:\Users\lukem\Desktop\Github AI Projects\Titanic\test.csv')

train.head()

#this is in the Initial-EDA branch


train.dtypes
# from this we can see that PassengerId, Survived, Pclass, SibSp, and Parch are dtype int64
# Fare is dtype float64
# Name, Sex, Ticket, Cabin, and Embarked are dtype object

train.isnull().sum()
# From this we can see that Cabin has 687 null values, Age has 177, and Embarked has 2
# Since Cabin has so many null values, I'm just going to remove that column
train = train.drop(['Cabin'], axis = 1)
test = test.drop(['Cabin'], axis = 1)

train.shape

# Replace the null age values with 21
train['Age'] = train['Age'].fillna(21)
test['Age'] = test['Age'].fillna(21)

# Replace the null Embarked values with S
train['Embarked'] = train['Embarked'].fillna('S')
train['Embarked'] = train['Embarked'].fillna('S')

# Replace the one null Fare value with 7
test['Fare'] = test['Fare'].fillna(7)

# Drop the name column, since I don't think name will have any useful information
train = train.drop(['Name'], axis = 1)
test = test.drop(['Name'], axis = 1)

# For the Sex class, I'm going to change this into a binary class, male = 1, female = 0
sex_dictionary = {'female':0, 'male':1}
train['Sex'] = train['Sex'].map(sex_dictionary)
test['Sex'] = test['Sex'].map(sex_dictionary)

