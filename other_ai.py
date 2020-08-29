import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


train = pd.read_csv(r'C:\Users\lukem\Desktop\Github AI Projects\Titanic\train.csv')
test = pd.read_csv(r'C:\Users\lukem\Desktop\Github AI Projects\Titanic\test.csv')
sample_submission = pd.read_csv(r'C:\Users\lukem\Desktop\Github AI Projects\Titanic\gender_submission.csv')

y = train.Survived
train = train.drop(['Survived'], axis = 1)
col_names = train.columns

# First impute the null values
imp_mean = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
imp_mean.fit(list(train.values)+list(test.values))
train_imputed = imp_mean.transform(train)
train_imputed = pd.DataFrame(train_imputed)
test_imputed = pd.DataFrame(imp_mean.transform(test))

train_imputed.columns, test_imputed.columns = col_names, col_names

train_imputed = train_imputed.set_index(['PassengerId'])
test_imputed = test_imputed.set_index(['PassengerId'])

# Next change the dyptes back to what they were
int_cols_dictionary = {'Pclass':'int32','Age':'int32','SibSp':'int32','Parch':'int32','Fare':'int32'}
train_converted = train_imputed.astype(int_cols_dictionary, copy=True)
test_converted = test_imputed.astype(int_cols_dictionary, copy=True)

# Next encode the columns with dtype 'object'
for f in train_converted.columns:
    if train_converted[f].dtype == "object" or test_converted[f].dtype == "object":
        le = preprocessing.LabelEncoder()
        le.fit(list(train_converted[f].values) + list(test_converted[f].values))
        train_converted[f] = le.transform(train_converted[f])
        test_converted[f] = le.transform(test_converted[f])
        
# Next, implement some sort of ai algorithm. We'll try lgb this time since I haven't done that one
X = train_converted
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
clf = lgb.LGBMClassifier()
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
acc = accuracy_score(y_test, pred)
print("The model is {model} and the accuracy is {accuracy:.2f}!!!".format(model = "lgb",accuracy = acc*100))
print("The model is {0} and the accuracy is {1:.2f}%".format("lgb", acc*100))
print("The model is {} and the accuracy is {:.2f}%%".format("lgb", acc*100))
print("The model is {model} and the accuracy is {accuracy:.2f}!!!".format(accuracy = acc*100, model = "lgb"))
print("The model is {} and the accuracy is {:%}".format("lgb", acc)) # Percentage format
print("The model is {} and the accuracy is {:+}".format("lgb", acc*100)) # Adds a plus sign
print("The model is {0} and the accuracy is {1:.4f}%".format("lgb",acc*100))





clf.fit(X, y)
pred = pd.DataFrame(clf.predict(test_converted))
frames = [sample_submission, pred]
submission = pd.concat(frames, axis = 1)
submission = submission.drop('Survived', axis = 1)
submission = submission.rename(columns = {0:'Survived'})

path = r'C:\Users\lukem\Desktop\Github AI Projects\Submissions\Titanic\ '
submission.to_csv(path + 'submission_lgb.csv', index = False)
