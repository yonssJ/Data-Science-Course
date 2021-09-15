import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from  pathlib import Path
from matplotlib.patches import Rectangle
from scipy import stats
import re
import time
import copy

from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import train_test_split, KFold,cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error 
from sklearn import linear_model, preprocessing, decomposition

from sklearn.pipeline import Pipeline
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler,  MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor
from sklearn import svm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import Lars

def var_target_features(df_numerical,emission_consomation):
    X_EU= copy.deepcopy(df_numerical)
    X_EU=X_EU.drop(emission_consomation,axis=1) 
    X_EU_features=X_EU.columns
 
    y_EU=copy.deepcopy(df_numerical)
    y_EU.drop(y_EU.columns.difference(emission_consomation), 1, inplace=True) 
    Y_EU_features=y_EU.columns

    print('CO2----->',X_EU_features)
    print('EU------>',Y_EU_features)
    return X_EU, y_EU, X_EU_features, Y_EU_features

def data_transform_normalisation(X_EU,scalar):
    std_scale_EU = scalar.fit(X_EU) 
    X_scale_EU   = std_scale_EU.transform(X_EU)
    return X_scale_EU

def plot_scatter_of_models(regressors,x_var,y_var,pred_train_base_NEW,eval_results_base_NEW,pred_train_base,fig,axs,label1,label2,plot_list,fir):
    ini =0
    ini0=0
    fig.subplots_adjust(left=0.1,bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
    for knx in regressors:
        data_brut_base=pred_train_base[knx]
        data_brut=pred_train_base_NEW[knx]
        eval_R = eval_results_base_NEW.T[knx]
    
        r2=eval_R['R Square Score']
        MAE=eval_R['MAE']
        Accuracy=eval_R['Précision']
        if fir==1:
            hs=sns.scatterplot(x=x_var, y=y_var, data=data_brut,ax=axs[ini0,ini],color='g', label = y_var).set(title=knx)
            hs1=sns.lineplot(x=x_var, y=x_var, data=data_brut_base,ax=axs[ini0,ini],color='k',label =x_var )
        
        else:
            hs=sns.scatterplot(x=x_var, y=y_var, data=data_brut,ax=axs[ini0,ini],color='g', label = label1).set(title=knx)
            hs0=sns.scatterplot(x=x_var, y=y_var, data=data_brut_base,ax=axs[ini0,ini],color='r', label =label2)
            hs1=sns.lineplot(x=x_var, y=x_var, data=data_brut_base,ax=axs[ini0,ini],color='k',label =x_var )
#
        axs[ini0,ini].set_ylim(data_brut_base[x_var].min(),data_brut_base[x_var].max())
        axs[ini0,ini].set_xlim(data_brut_base[x_var].min(),data_brut_base[x_var].max())
    
        axs[ini0,ini].text(.60,0.48,'R$^2$  =' + str(round(r2,2)),horizontalalignment='left',
                       transform=axs[ini0,ini].transAxes)
        axs[ini0,ini].text(.60,0.4,'MAE =' + str(round(MAE,2)),horizontalalignment='left',
                       transform=axs[ini0,ini].transAxes)
        axs[ini0,ini].text(.60,0.32,'Accuracy =' + str(round(Accuracy,2)),horizontalalignment='left',
                       transform=axs[ini0,ini].transAxes)
        ini=ini+1
        if ini==plot_list:
            ini0=ini0+1
            ini =0
                    
def Multi_output_evaluate_model(regressors,X_train, y_train,X_test, y_test,seed):

    results= pd.DataFrame(index=["R Square Score", "test_rmse", "train_rmse", "variance",'Model ameliorasion','run_time','MAE','Précision'])  
    
    Findex=['Cible predite']    
    frames_test = [] 
    frames_train= [] 
    column_regressor=[]
    
    for ks in regressors:
        column_regressor.append(ks)
#----------------------------------------------------------------

    for model_idx in regressors:
        
        start_time = time.time()
        #
        print(model_idx)
        regressor  = regressors[model_idx]
        
        regressor.fit(X_train, y_train)
        y_train_pred   = regressor.predict(X_train)
        y_test_pred    = regressor.predict(X_test)
              
        test_rmse =np.sqrt(mean_squared_error(y_test,y_test_pred))
        train_rmse=np.sqrt(mean_squared_error(y_train,y_train_pred))
        
        variance =abs(train_rmse - test_rmse)
        MAE      =100*mean_absolute_error(y_test,y_test_pred)   
        Precision=100-MAE
                                 
        coef_det = r2_score(y_train, y_train_pred)
        
        if test_rmse<train_rmse:
            good='T'
        else:
            good='F'
   
        test_pred= pd.DataFrame(y_test_pred,index=y_test.index,columns=Findex)
        train_pred=pd.DataFrame(y_train_pred,index=y_train.index,columns=Findex)
        
        run_time=format(round((time.time() - start_time),2))  
        
        results[model_idx]=[coef_det,test_rmse, train_rmse, variance,good,run_time,MAE,Precision]
        
        Y_pred_test = pd.concat([y_test, test_pred],axis=1)
        Y_pred_train= pd.concat([y_train,train_pred],axis=1)
        
        frames_train.append(Y_pred_train)
        frames_test.append(Y_pred_test)
 
    df_pred_test = pd.concat(frames_test, keys=column_regressor, axis=1)
    df_pred_train = pd.concat(frames_train, keys=column_regressor, axis=1)
    
    results=results.T
    results=results.sort_values(by=["test_rmse"],axis=0, ascending=True)
    return results, df_pred_test, df_pred_train

def minmax_norm(df):
    return (df - df.min()) / ( df.max() - df.min())

def best_model_find(grid_search,X_train,y_train):
    
    best_estim=grid_search.best_estimator_
    best_score=grid_search.best_score_
    best_param=grid_search.best_params_
    best_time=grid_search.refit_time_
    #feature_importances=grid_search.feature_importances_
    best_estim.fit(X_train,y_train)
    y_train_pred= best_estim.predict(X_train)
       
    train_rmse=np.sqrt(mean_squared_error(y_train,y_train_pred))
    MAE_tr      =100*mean_absolute_error(y_train,y_train_pred)   
    Precision_tr=100-MAE_tr
    coef_det = r2_score(y_train, y_train_pred)
     
    return y_train_pred, coef_det, train_rmse,best_score,best_estim,best_param,MAE_tr,Precision_tr,best_time
       
def Multi_output_improved_regressor_model(regressors,params_regressors,X_train, y_train,n_splits,seed):
    
    results= pd.DataFrame(index=["R Square Score" ,"RMSE",\
                                "best_score_","best_estimator_","best_params_",'run_time','MAE','Précision',"best_time"]) 
    
    Findex=['Cible predite']    
    frames_train= [] 
    column_regressor=[]
    
    for ks in regressors:
        column_regressor.append(ks)
#----------------------------------------------------------------y_EU

    for model_idx in regressors:
        
        start_time = time.time()
        #
        print(model_idx)
        regressor  = regressors[model_idx]
        param_grid= params_regressors[model_idx]
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        dfs=kfold
        cv_results = cross_val_score(regressor , X_train, y_train,cv=kfold)
    
        grid_search = GridSearchCV(estimator=regressor, param_grid=param_grid, cv=dfs)
        
        grid_search.fit(X_train, y_train)
        
        [y_train_pred,coef_det,train_rmse,best_score,best_estim,best_param,MAE_tr,Precision_tr,best_time]=\
                        best_model_find(grid_search,X_train,y_train)
                
        train_pred=pd.DataFrame(y_train_pred,index=y_train.index,columns=Findex)
        
        run_time=format(round((time.time() - start_time),2))  
    
        results[model_idx]=[coef_det, train_rmse,best_score,best_estim,best_param,run_time,MAE_tr,Precision_tr,best_time]
    
        Y_pred_train= pd.concat([y_train,train_pred],axis=1)
        frames_train.append(Y_pred_train)

    df_pred_train = pd.concat(frames_train, keys=column_regressor, axis=1)
    
    results=results.T
    
    results=results.sort_values(by=["RMSE"],axis=0, ascending=True)
    return results,df_pred_train

def boxplot_plot(df, varname,xx,yy,valuevar,place,linewidthy,xlabell,miny,maxy,melt):
    if melt==1:
        gfg = sns.boxplot(x=xx, y=yy, data=df[df[varname]==valuevar],ax=place,linewidth=linewidthy)
        gfg.set(xlabel = xlabell, ylabel = valuevar)
        plt.rcParams['font.size'] = '18'
    else:
        gfg = sns.boxplot(x=xx, y=yy, data=df,ax=place,linewidth=linewidthy)
        gfg.set(xlabel = xlabell, ylabel = valuevar)
        gfg.set(ylim=(miny, maxy))
        plt.rcParams['font.size'] = '18'


def plot_scatter_of_models_3(regressors,x_var,y_var,pred_train_base_NEW,eval_results_base_NEW,pred_train_base,fig,axs,label1,label2,plot_list,fir):
    ini =0
    ini0=0
    fig.subplots_adjust(left=0.1,bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
    for knx in regressors:
        data_brut_base=pred_train_base[knx]
        data_brut=pred_train_base_NEW[knx]
        eval_R = eval_results_base_NEW.T[knx]
    
        r2=eval_R['R Square Score']
        MAE=eval_R['MAE']
        Accuracy=eval_R['Précision']
        if fir==1:
            hs=sns.scatterplot(x=x_var, y=y_var, data=data_brut,ax=axs[ini0],color='g', label = y_var).set(title=knx)
            hs1=sns.lineplot(x=x_var, y=x_var, data=data_brut_base,ax=axs[ini0],color='k',label =x_var )
        
        else:
            hs=sns.scatterplot(x=x_var, y=y_var, data=data_brut,ax=axs[ini0],color='g', label = label1).set(title=knx)
            hs0=sns.scatterplot(x=x_var, y=y_var, data=data_brut_base,ax=axs[ini0],color='r', label =label2)
            hs1=sns.lineplot(x=x_var, y=x_var, data=data_brut_base,ax=axs[ini0],color='k',label =x_var )
#
        axs[ini0].set_ylim(data_brut_base[x_var].min(),data_brut_base[x_var].max())
        axs[ini0].set_xlim(data_brut_base[x_var].min(),data_brut_base[x_var].max())
    
        axs[ini0].text(.60,0.48,'R$^2$  =' + str(round(r2,2)),horizontalalignment='left',
                       transform=axs[ini0].transAxes)
        axs[ini0].text(.60,0.4,'MAE =' + str(round(MAE,2)),horizontalalignment='left',
                       transform=axs[ini0].transAxes)
        axs[ini0].text(.60,0.32,'Accuracy =' + str(round(Accuracy,2)),horizontalalignment='left',
                       transform=axs[ini0].transAxes)

        ini0=ini0+1
         