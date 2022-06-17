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
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt 


df= pd.read_csv('sample_train.csv')
pd.options.display.max_columns = None  
df=df.drop(columns='emplid')

#train validation split : The orginal train sample broken to train and validation for hyper parameter optimization
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(df.iloc[:,:29]
                                                    , df['label'] , test_size=0.3, random_state=42, stratify=df['label'])
##
scaler = MinMaxScaler(feature_range=[0, 1])
x_train=scaler.fit_transform(x_train)
x_test=scaler.fit_transform(x_test)


#baseline xgb
clf_xgb_base = xgb.XGBClassifier(objective='binary:logistic', seed=42)
clf_xgb_base.fit(x_train,y_train,
                 verbose=True,
                 early_stopping_rounds=10,
                 eval_metric='aucpr',
                 eval_set=[(x_test,y_test)])
clf_xgb_base.predict(x_test)
plot_confusion_matrix(clf_xgb_base,x_test,y_test,
                      display_labels=["left", "persisted"])

#optimizing the XGBoost model in 3 step grid search to find the boundaries
import multiprocessing
n_cpus = multiprocessing.cpu_count()
#round 1 of cv 
from joblib import parallel_backend

with parallel_backend('threading', n_jobs=n_cpus-10)
#round 1 of cv 
from joblib import parallel_backend

with parallel_backend('threading', n_jobs=n_cpus-10):
    param_grid={
            'max_depth':[3,5,7,12],
            'learning_rate':[1, 0.1,0.01,0.05],
            'gamma':[0,0.25,1],
            'reg_lambda':[0,1,10],
            'scale_pos_weight':[0.4,0.5,1,2,3]
            
    }

    optimal_params=GridSearchCV(estimator=xgb.XGBClassifier(objective='binary:logistic', seed=42,subsample=0.9, cosample_bytree=0.5),
                           param_grid=param_grid, scoring='roc_auc',verbose=3,
                           n_jobs=n_cpus-10, cv=3
                          )
    optimal_params.fit(x_train,y_train,
                   early_stopping_rounds=10,
                   eval_metric='auc',
                   eval_set=[(x_test,y_test)], verbose=3)
    print(optimal_params.best_params_)

param_grid={
            'max_depth':[3,5,7,12],
            'learning_rate':[1, 0.1,0.01,0.05],
            'gamma':[0,0.25,1],
            'reg_lambda':[0,1,10],
            'scale_pos_weight':[0.4,0.5,1,2,3]
            
    }

optimal_params=GridSearchCV(estimator=xgb.XGBClassifier(objective='binary:logistic', seed=42,subsample=0.9, cosample_bytree=0.5),
                           param_grid=param_grid, scoring='roc_auc',verbose=3,
                           cv=3
                          )
optimal_params.fit(x_train,y_train,
                   early_stopping_rounds=10,
                   eval_metric='auc',
                   eval_set=[(x_test,y_test)], verbose=3)
print(optimal_params.best_params_)
param_grid={
            'max_depth':[3,5,7,12],
            'learning_rate':[1, 0.1,0.01,0.05],
            'gamma':[0,0.25,1],
            'reg_lambda':[0,1,10],
            'scale_pos_weight':[0.4,0.5,1,2,3]
            
    }

optimal_params=dcv.GridSearchCV(estimator=xgb.XGBClassifier(objective='binary:logistic', seed=42,subsample=0.9, cosample_bytree=0.5),
                           param_grid=param_grid, scoring='roc_auc',
                           cv=3
                          )
optimal_params.fit(x_train,y_train,
                   early_stopping_rounds=10,
                   eval_metric='auc',
                   eval_set=[(x_test,y_test)])
print(optimal_params.best_params_)

results= pd.DataFrame(optimal_params.cv_results_)
clf_xgb_optimized.fit(x_test,y_test)

                      display_labels=["left", "persisted"])
plot_confusion_matrix(clf_xgb_optimized,x_test,y_test,
                      display_labels=["left", "persisted"])

xgb.plot_importance(clf_xgb_optimized)
plt.figure(figsize = (100,100))
plt.show()

#drawing feature importance
feature_importance=pd.DataFrame()
feature_importance['feature']=feature_name
feature_importance['importance']=clf_xgb_optimized.feature_importances_
feature_importance= feature_importance.loc[feature_importance['feature']!='censusSemNo']
fig_dims = (24, 24)
fig, ax = plt.subplots(figsize=fig_dims)
sns.barplot(
    y='feature', 
    x='importance', 
    color='#69b3a2', data=feature_importance.sort_values(by=['importance'],ascending=False),ax=ax)
