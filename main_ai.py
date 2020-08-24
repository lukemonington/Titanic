import pandas as pd
pd.set_option('max_columns', None)
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


# Get train and test dataset
train = pd.read_csv(r'C:\Users\lukem\Desktop\Github AI Projects\Titanic\train.csv')
test = pd.read_csv(r'C:\Users\lukem\Desktop\Github AI Projects\Titanic\test.csv')
example_submission = pd.read_csv(r'C:\Users\lukem\Desktop\Github AI Projects\Titanic\gender_submission.csv')
PassIDs = test['PassengerId']

target = train['Survived']
train = train.drop('Survived', axis = 1)

train = train.fillna(-999)
test = test.fillna(-999)

X_train = train
X_test = test
y_train = target

# For loop to encode all columns with "object" datatype
for f in X_train.columns:
    if X_train[f].dtype == "object" or X_test[f].dtype == "object":
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(X_train[f].values) + list(X_test[f].values))
        X_train[f] = lbl.transform(list(X_train[f].values))
        X_test[f] = lbl.transform(list(X_test[f].values))

X = X_train
y = y_train
test = X_test

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2)

def get_mae(max_leaf_nodes, X_train, X_val, y_train, y_val):
    



clf = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=9,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    missing=-999,
    random_state=2019,
    tree_method='gpu_hist'  # THE MAGICAL PARAMETER
)
%time clf.fit(X_train, y_train)


pred = clf.predict(X_test)


submission = pd.DataFrame(PassIDs, columns = ['PassengerId'])
submission['Survived'] = pred
path = r"C:\Users\lukem\Desktop\Github AI Projects\Submissions\Titanic\ "
submission.to_csv(path + "submissionv1.csv", index = False)
