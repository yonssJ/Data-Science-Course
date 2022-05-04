import pandas as pd
from sqlalchemy import create_engine

def avail_client(data):
    todelete='Unnamed: 0' 
    def remove_Unnamed(df,todelete):
        for i in df.columns:
            if todelete in i:
                df=df.drop([i],axis=1)
            return df
    df=pd.read_csv(data)
    df=remove_Unnamed(df,todelete)
    df["client_id"]=df["SK_ID_CURR"]
    #df["client_name"]=df["SK_ID_CURR"]
    df=df.drop("SK_ID_CURR",axis=1)
    clientList=df['client_id'].values

    return clientList,df





