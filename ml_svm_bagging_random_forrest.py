import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from zipfile import ZipFile
from io import StringIO
import fnmatch
from pandas import ExcelWriter
from dirty_cat import GapEncoder
from sklearn.preprocessing import OneHotEncoder
import scipy.sparse
from sklearn.compose import ColumnTransformer
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

#read data

df =pd.read_csv('sample.csv')

#split
#train test split 
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(df.iloc[:,:29]
                                                    , df['label'] , test_size=0.3, random_state=42, stratify=df['label'])

scaler = MinMaxScaler(feature_range=[0, 1])
x_train=scaler.fit_transform(x_train)
x_test=scaler.fit_transform(x_test)

#grid search: parallel hyperparametertunning
#this acceleratess
import multiprocessing
n_cpus = multiprocessing.cpu_count()
#round 1 of cv 
from joblib import parallel_backend
with parallel_backend('threading', n_jobs=n_cpus-10)


#round 1 of cv 
from joblib import parallel_backend

with parallel_backend('threading', n_jobs=n_cpus-10):
    param_grid = [
              {'C': [1, 10, 100, 1000], 'gamma': [0.0001, 0.01,1,10,50], 'kernel': ['rbf']},
              {'C': [1, 10, 100, 1000], 'degree': [0.1, 1, 2, 3, 5, 10], 'kernel': ['poly']}
    ]
    weights = {0:0.15, 1:0.85}
    svm=SVC(class_weight=weights)

    grid = GridSearchCV(svm, param_grid, refit = False, verbose = 3, scoring = 'accuracy')
  
    # fitting the model for grid search
    grid.fit(x_train, y_train)

    # print best parameter after tuning
    print(grid.best_params_)
  
    # print how our model looks after hyper-parameter tuning
    print(grid.best_estimator_)

    grid_predictions = grid.predict(x_test)
  
    # print classification report
    print(classification_report(y_test, grid_predictions))
    
    

#random forests
weights = {0:0.15, 1:0.85}
from sklearn.ensemble import RandomForestClassifier
m = RandomForestClassifier(n_estimators=1000,class_weight="balanced", n_jobs=-1)
m.fit(x_train,y_train)
y_pred = m.predict(x_test) #Test/Validating the model
y_pred_train=m.predict(x_train)
print('Train Accuracy: %.2f' % accuracy_score(y_train, y_pred_train))
print('Test Accuracy: %.2f' % accuracy_score(y_test, y_pred))

#bagging classifier
from sklearn.ensemble import BaggingClassifier
model = BaggingClassifier(base_estimator=m,n_estimators=10, random_state=0)
# define evaluation procedure
# cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
model.fit(x_train,y_train)
y_pred = model.predict(x_test) #Test/Validating the model
y_pred_train=model.predict(x_train)
print('Train Accuracy: %.2f' % accuracy_score(y_train, y_pred_train))
print('Test Accuracy: %.2f' % accuracy_score(y_test, y_pred))
    
    
    
