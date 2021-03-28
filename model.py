#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, StratifiedKFold
import pickle

import os
os.chdir(r"") 

df = pd.read_csv('UniversalBank.csv')

# We will find the mean of positive experience values for above ages and use it to replace

# having -1 exp
a = df[df['Experience'] == -1]['Age'].value_counts().index.tolist()
x = df[df['Experience'] == -1]['Experience'].index.tolist()
for i in x:
    df.loc[i,'Experience'] = df[(df['Age'].isin(a)) & (df.Experience > 0)].Experience.mean()
    
    
# having -2 exp
b = df[df['Experience'] == -2]['Age'].value_counts().index.tolist()
y = df[df['Experience'] == -2]['Experience'].index.tolist()
for i in y:
    df.loc[i,'Experience'] = df[(df['Age'].isin(b)) & (df.Experience > 0)].Experience.mean()
    
    
# having -3 exp
c = df[df['Experience'] == -3]['Age'].value_counts().index.tolist()
z = df[df['Experience'] == -3]['Experience'].index.tolist()
for i in z:
    df.loc[i,'Experience'] = df[(df['Age'].isin(c)) & (df.Experience > 0)].Experience.mean()

df.drop(columns=['ZIP Code','ID'], inplace=True)


#rearranging columns

df = df.loc[:,['Age', 'Experience', 'Income', 'Education', 'Family', 'CreditCard', 'CCAvg', 'Online',
       'Mortgage', 'Securities Account', 'CD Account','Personal Loan']]

#We will use experience column and not use age column as both are highly correlated

x = df.iloc[:,1:11]
y = df.iloc[:,-1]


from sklearn.ensemble import GradientBoostingClassifier
# parameter grid
param_grid = {"learning_rate": [0.3, 0.6, 0.9],
              "subsample": [0.3, 0.6, 0.9],
              "max_depth": [3,6,9],
              "max_features": [3,6,9],
              "min_samples_leaf": range(1, 5),
              "min_samples_split": [3,6,9],
              "random_state": [5]
             }


GBC = GradientBoostingClassifier(max_depth=2, n_estimators=200)


# run grid search
folds = 10
grid_search_GBC = GridSearchCV(GBC, 
                               cv = folds,
                               param_grid=param_grid, 
                               return_train_score=True,                         
                               verbose = 1,
                               scoring = 'recall',
                               n_jobs= -1)


grid_search_GBC.fit(x,y)

GBC = grid_search_GBC.best_estimator_

GBC.fit(x, y)

pickle.dump(GBC, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[10, 120, 2, 2, 1, 3, 1, 0, 0, 1]]))