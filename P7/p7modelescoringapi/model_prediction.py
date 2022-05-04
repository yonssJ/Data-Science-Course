import pickle
import numpy as np

def remove_Unnamed(df,todelete):
    for i in df.columns:
        if todelete in i:
            df=df.drop([i],axis=1)
        return df

def prediction(user,X_test,model_file,threshold):
#----
    todelete='Unnamed: 0'  

    X_test=remove_Unnamed(X_test,todelete)
    X_test=X_test.loc[X_test['client_id'] == user]
    X_test=X_test.drop('client_id',axis=1)
    loaded_model = pickle.load(open(model_file, 'rb'))
    yy_pred = (loaded_model.predict_proba(X_test)[:,1] >= threshold).astype('float64')
    yy_pred = np.array(yy_pred > 0) * 1

    return yy_pred
