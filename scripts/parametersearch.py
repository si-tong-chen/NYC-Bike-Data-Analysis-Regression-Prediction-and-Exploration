#!/usr/bin/env python
# coding: utf-8

# In[ ]:


### Hyperparameter search

from tqdm import tqdm
import random
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import xgboost as xgb
import numpy as np

def Hyperparameter_search(num_searches,X_train,y_train,X_test,y_test):
    n_estimators = range(40,500,3)
    max_depth = range(3,8)
    subsample= np.arange(0.1,1,0.1)
    learning_rate=np.arange(0.1, 0.51, 0.01)
    colsample_bytree=np.arange(0.1,1,0.1)


    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)


    num_searches = num_searches
    best_score = -float("inf")  
    best_params = {}  
    r2_record = []
    params_recod = []

    for _ in tqdm(range(num_searches), desc="Hyperparameter Search"):
        est = random.choice(n_estimators)
        depth = random.choice(max_depth)
        sub = random.choice(subsample)
        lr =  random.choice(learning_rate)
        colsample = random.choice(colsample_bytree)

        model = xgb.XGBRegressor(objective='reg:squarederror',
                                 n_jobs=-1,
                                 n_estimators=est, 
                                 max_depth=depth, 
                                 subsample=sub, 
                                 learning_rate=lr, 
                                 colsample_bytree=colsample,
                                 random_state=2023, 
                                 max_features=None, 
                                 alpha=0.8)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        r2_record.append(r2)
        best_params = {
            'n_estimators': est,
            'max_depth': depth,
            'subsample': sub,
            'learning_rate': lr,
            'colsample_bytree': colsample
        }
        params_recod.append(best_params)



    print("the best score:", max(r2_record))
    print("Hyperparameter:", params_recod[r2_record.index(max(r2_record))])

