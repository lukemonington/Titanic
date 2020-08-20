import pandas as pd
pd.set_option('max_columns', None)
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score


# Get train and test dataset
train = pd.read_csv(r'C:\Users\lukem\Desktop\Github AI Projects\Titanic\train.csv')
test = pd.read_csv(r'C:\Users\lukem\Desktop\Github AI Projects\Titanic\test.csv')
example_submission = pd.read_csv(r'C:\Users\lukem\Desktop\Github AI Projects\Titanic\gender_submission.csv')
PassIDs = test['PassengerId']

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

# For the embarked column, there seems to be two rows with a value of "", replace that with "S" (most common)
print(train['Embarked'].value_counts().sort_values())
train['Embarked'] = train['Embarked'].replace("", "S")
print(test['Embarked'].value_counts().sort_values())

train.shape
# Train shape = (891, 10)
test.shape
# Test shape = 418, 9

target_df = train['Survived']
train = train.drop(['Survived'], axis = 1)

train_test_list = [train,test]
train_test_df = pd.concat(train_test_list, axis = 0)

train_test_dummies = train_test_df['Embarked']
train_test_df = train_test_df.drop(['Embarked'], axis = 1)

dummy_df = pd.get_dummies(train_test_dummies, sparse = False)
dummies_list = [train_test_df, dummy_df]
train_test_df = pd.concat(dummies_list, axis = 1)

train = train_test_df.iloc[0:891,:]
test = train_test_df.iloc[891:,:]

train = train.drop(['Ticket'], axis = 1)
test = test.drop(['Ticket'], axis = 1)
seed = 7
test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(train, target_df, test_size=test_size, random_state=seed)

# Using data to train AI
model = XGBClassifier(
    learning_rate = 0.1,
    n_estimators = 500,
    max_depth = 7,
    min_child_weight = 0.5)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

pred = model.predict(test)
# Creating submission csv file
submission = pd.DataFrame(PassIDs, columns = ['id'])
submission['target'] = pred
path = r"C:\Users\lukem\Desktop\Github AI Projects\Titanic\submissions\titanic_submission_1.csv"
submission.to_csv(path, index = False)
