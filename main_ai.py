import pandas as pd
pd.set_option('max_columns', None)
import numpy as np


# Get train and test dataset
train = pd.read_csv(r'C:\Users\lukem\Desktop\Github AI Projects\Titanic\train.csv')
test = pd.read_csv(r'C:\Users\lukem\Desktop\Github AI Projects\Titanic\test.csv')

train.head()

