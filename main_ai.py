import pandas as pd
pd.set_option('max_columns', None)
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer


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
    dtr = DecisionTreeRegressor(max_leaf_nodes = max_leaf_nodes)
    dtr.fit(X_train, y_train)
    preds = dtr.predict(X_val)
    mae = mean_absolute_error(preds, y_val)
    return(mae)

def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)

model = DecisionTreeRegressor(max_leaf_nodes=20, random_state=0)
model.fit(X_train, y_train)
preds_val = model.predict(X_val)
mae = mean_absolute_error(y_val, preds_val)
print(mae)

train_X = X_train
train_y = y_train
val_X = X_val
val_y = y_val
forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(train_X, train_y)
melb_preds = forest_model.predict(val_X)
print(mean_absolute_error(val_y, melb_preds))

leafs_to_test = [2, 20, 50, 500]

for i in leafs_to_test:
    my_mae = get_mae(i, X_train, X_val, y_train, y_val)
    print("max_leaf_nodes: %d, mean absolute error: %d" %(i, my_mae))


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
