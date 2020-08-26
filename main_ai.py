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
import eli5
from eli5.sklearn import PermutationImportance
from sklearn.feature_selection import SelectFromModel


# Get train and test dataset
X_train = pd.read_csv(r'C:\Users\lukem\Desktop\Github AI Projects\Titanic\train.csv')
X_test = pd.read_csv(r'C:\Users\lukem\Desktop\Github AI Projects\Titanic\test.csv')
example_submission = pd.read_csv(r'C:\Users\lukem\Desktop\Github AI Projects\Titanic\gender_submission.csv')

PassIDs = X_test['PassengerId']
y_train = X_train['Survived']
X_train = X_train.drop('Survived', axis = 1)
train_cols = X_train.columns


imp = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent', copy = True)
X_train = pd.DataFrame(imp.fit_transform(X_train))
X_test = pd.DataFrame(imp.transform(X_test))

X_train.columns = train_cols
X_test.columns = train_cols


X_train.set_index('PassengerId', inplace = True)
X_test.set_index('PassengerId', inplace = True)

# For loop to encode all columns with "object" datatype
for f in X_train.columns:
    if X_train[f].dtype == "object" or X_test[f].dtype == "object":
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(X_train[f].values) + list(X_test[f].values))
        X_train[f] = lbl.transform(list(X_train[f].values))
        X_test[f] = lbl.transform(list(X_test[f].values))

X_train = X_train[['Sex','Age','Pclass','Parch','SibSp','Cabin']]
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.2)

def test_dtregressor(max_leaf_nodes, X_train, X_val, y_train, y_val):
    dtr = DecisionTreeRegressor(max_leaf_nodes = max_leaf_nodes, random_state = 2020)
    dtr.fit(X_train, y_train)
    preds = dtr.predict(X_val)
    mae = mean_absolute_error(preds, y_val)
    return(mae)

def test_rfregressor(n_estimators, X_train, X_val, y_train, y_val):
    rf = RandomForestRegressor(n_estimators = n_estimators, random_state = 2020)
    rf.fit(X_train, y_train)
    preds = rf.predict(X_val)
    mae = mean_absolute_error(preds, y_val)
    return(mae)


# test different values for random forest
estimators_to_test = [50, 100, 250, 500]
my_mae = list()
for i in estimators_to_test:
    my_mae.append(test_rfregressor(i, X_train, X_val, y_train, y_val))
    print(f"n_estimators: {i}, mean_absolute_error: {my_mae[-1]}")

# test different values for decision tree
leafs_to_test = [2, 20, 50, 500]
my_mae = list()
for i in leafs_to_test:
    my_mae.append(test_dtregressor(i, X_train, X_val, y_train, y_val))
    print(f"max_leaf_nodes: {i}, mean absolute error: {my_mae[-1]}")

final_dtr = DecisionTreeRegressor(max_leaf_nodes = 500, random_state = 2020)
final_dtr.fit(X_train, y_train)
preds = final_dtr.predict(X_val)

# Trying out permutation importance 
# Permutation importance asks: If I randomly shuffle a single column of the validation data, leaving the target and all
# other columns in place, how would that affect the accuracy of predictions in that now-shuffled data?
perm = PermutationImportance(final_dtr, random_state=1).fit(X_val, y_val)
# Store feature weights in an object
html_obj = eli5.show_weights(perm, feature_names = X_val.columns.tolist())

# Write html object to a file (adjust file path; Windows path is used here)
with open(r'C:\Users\lukem\Desktop\Github AI Projects\Titanic\titanic-importance.htm','wb') as f:
    f.write(html_obj.data.encode("UTF-8"))
    
# I think this allows me to select only the columns that are above a certain threshhold
sel = SelectFromModel(perm, threshold=0.04, prefit=True)
X_trans = sel.transform(X_train)




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
submission.to_csv(path + "trying-out-imputation-submissionv2.csv", index = False)
