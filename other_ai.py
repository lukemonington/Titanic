import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
from matplotlib import pyplot as plt
from pdpbox import pdp, get_dataset, info_plots


train = pd.read_csv(r'C:\Users\lukem\Desktop\Github AI Projects\Titanic\train.csv')
test = pd.read_csv(r'C:\Users\lukem\Desktop\Github AI Projects\Titanic\test.csv')
sample_submission = pd.read_csv(r'C:\Users\lukem\Desktop\Github AI Projects\Titanic\gender_submission.csv')

#Survived dataframe
Survived = train.loc[train.Survived == 1]

#Didn't survive dataframe
Dead = train.loc[train.Survived == 0]

# Dead and in Pclass 1
Dead_Pclass1 = train.loc[(train.Survived==0) & (train.Pclass==1)]

y = train.Survived
train = train.drop(['Survived'], axis = 1)
col_names = train.columns



sns.set(style="darkgrid")
ax = sns.barplot(x = y.value_counts().index, y = y.value_counts())

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


# Trying out a partial dependence plot
# Create the data that we will plot
pdp_Pclass = pdp.pdp_isolate(model = clf, dataset = X_test, model_features = X.columns, feature = 'Pclass')

# plot it
pdp.pdp_plot(pdp_Pclass, 'Pclass')
plt.show()

pdp_Age = pdp.pdp_isolate(model = clf, dataset = X_test, model_features = X.columns, feature = 'Age')
pdp.pdp_plot(pdp_Age, 'Age')
plt.show()

pdp_Sex = pdp.pdp_isolate(model = clf, dataset = X_test, model_features = X.columns, feature = 'Sex')
pdp.pdp_plot(pdp_Sex, 'Sex')
plt.show()

# Trying out a 2D partial dependence plot
# Similar to previous PDP plot except we use pdp_interact instead of pdp_isolate and pdp_interact_plot instead of pdp_isolate_plot
features_to_plot = ['Age', 'Sex']
inter1  =  pdp.pdp_interact(model = clf, dataset=X_test, model_features=X.columns, features=features_to_plot)
# Initially had some problems with plot_type = 'contour', but I had no problems with plot_type = 'grid'
pdp.pdp_interact_plot(pdp_interact_out=inter1, feature_names=features_to_plot, plot_type='grid')
plt.show()


y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("The model is {model} and the accuracy is {accuracy:.2f}!!!".format(model = "lgb",accuracy = acc*100))
print("The model is {0} and the accuracy is {1:.2f}%".format("lgb", acc*100))
print("The model is {} and the accuracy is {:.2f}%%".format("lgb", acc*100))
print("The model is {model} and the accuracy is {accuracy:.2f}!!!".format(accuracy = acc*100, model = "lgb"))
print("The model is {} and the accuracy is {:%}".format("lgb", acc)) # Percentage format
print("The model is {} and the accuracy is {:+}".format("lgb", acc*100)) # Adds a plus sign
print("The model is {0} and the accuracy is {1:.4f}%".format("lgb",acc*100))

cm = confusion_matrix(y_test, y_pred)
print("The model is {0} and the accuracy is {1:.4f}%".format("lgb",acc*100))
#print(cm)
print("The number of true positives is {}".format(cm[0,0]))
print("The number of false positives is {}".format(cm[0,1]))
print("The number of true negatives is {}".format(cm[1,1]))
print("The number of false negatives is {}".format(cm[1,0]))

# visualize confusion matrix with seaborn heatmap
cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'], 
                                 index=['Predict Positive:1', 'Predict Negative:0'])
sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')


clf.fit(X, y)
pred = pd.DataFrame(clf.predict(test_converted))
frames = [sample_submission, pred]
submission = pd.concat(frames, axis = 1)
submission = submission.drop('Survived', axis = 1)
submission = submission.rename(columns = {0:'Survived'})

path = r'C:\Users\lukem\Desktop\Github AI Projects\Submissions\Titanic\ '
submission.to_csv(path + 'submission_lgb.csv', index = False)
