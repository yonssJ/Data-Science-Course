import pandas as pd
import plotly.graph_objects as go
from copy import deepcopy
import dash_bootstrap_components as dbc
from catboost import CatBoostClassifier
import shap
import pickle

import numpy as np

model = CatBoostClassifier()
selected_model='C.Pred_proba_CatBoostClassifier'
model_file = 'data/finalized_model.sav'
X_knn = 'data/X_train_random_sample_5000.csv'
Y_knn = 'data/Y_train_random_sample_5000.csv'
todelete='Unnamed: 0'  

def remove_Unnamed(df,todelete):
    for i in df.columns:
        if todelete in i:
            df=df.drop([i],axis=1)
        return df
#----
df_predite =pd.read_csv('data/df_pred_over.csv')
Y_test=df_predite['TARGET']
Y_pred=df_predite[selected_model]
df_predite =df_predite.drop(df_predite.columns.difference([selected_model]), axis=1)
#----
df=pd.read_csv('data/persodata_client_test.csv')
df=remove_Unnamed(df,todelete)

df1=pd.merge(df, df_predite, left_index=True, right_index=True)
df1=pd.merge(df1, Y_test, left_index=True, right_index=True)
df1=df1.set_index('SK_ID_CURR')
small=1
if small==1:
    sampling=1000
    seed=1
    df1=df1.sample(n=sampling, random_state=seed)
else:
    print("Work with all data")

available_clients=  df1.index

todrop=['CODE_GENDER','NAME_FAMILY_STATUS','OCCUPATION_TYPE','NAME_INCOME_TYPE','ORGANIZATION_TYPE']
dataperso=df1[todrop]
df1=df1.drop(todrop, axis=1)
options = []
for column in available_clients.tolist(): 
    options.append({'label': '{}'.format(column, column), 'value': column})

options = options[1:]
X_samples=deepcopy(df1)
X_samples=df1.reset_index()
Y_sample=X_samples["TARGET"]
X_sample=X_samples.drop([selected_model,'SK_ID_CURR','TARGET'], axis=1)
#----
loaded_model = pickle.load(open(model_file, 'rb'))
explainer     = shap.TreeExplainer(loaded_model)
shap_values   = explainer.shap_values(X_sample)
shap_values_df= pd.DataFrame(shap_values,columns=X_sample.columns) 

X_train_sample=pd.read_csv(X_knn)
Y_train_sample=pd.read_csv(Y_knn)
X_train_sample=remove_Unnamed(X_train_sample,todelete)
Y_train_sample=remove_Unnamed(Y_train_sample,todelete)

