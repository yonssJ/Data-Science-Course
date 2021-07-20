import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pathlib
from pandas import Series

def remove_outlier(df_in,q1value,q3value):
    import copy
    df_out = copy.deepcopy(df_in)
    
    q1 =df_in.quantile(q1value)
    q3 =df_in.quantile(q3value)
    iqr = q3-q1  
    fence_low  = q1-1.5*iqr
    fence_high = q3+1.5*iqr
    #===== Remove
    df_out = df_in[~((df_in < fence_low) |(df_in > fence_high)).any(axis=1)]
    return df_out

def isnull_values_sum(df):
    taille = len(df)/2
    null   = df.isnull().sum() 
    b=null.le(taille)
    cols_to_keep = df.columns[b]
    df = df.loc[:, cols_to_keep]
    return df

def plot_box_plot(df,xlabel_name,ylabel_name):
    color = dict(boxes='black', whiskers='black', medians='red', caps='black')
    ax = df.plot.box(color=color,whiskerprops = dict(linestyle='-',linewidth=2.0, color='black'),figsize=(24, 8),fontsize=18)
    ax.set_xlabel(xlabel_name,fontsize=18)
    ax.set_ylabel(ylabel_name,fontsize=18)
    plt.grid(color='k', linestyle='-', linewidth=.1)
    plt.xticks(rotation=90)
    ax.label_outer() 

    plt.savefig(figure_name,format='jpeg',dpi=100,bbox_inches='tight')

def read_csv_morceaux(datapath,linein,lineend):
    df = pd.read_csv(datapath,
                     sep="\t", encoding="utf-8",skiprows=linein, nrows=lineend, low_memory=False)
# Pour enlever les NaN que sont dans 'countries_en' e remplacer par  np.nan
    df = df.replace('NAN', np.nan) 
    
# Selectioner seulement les donnees du marche Français vu que le projet est pour la France
    mask = df[country].str.contains(rpays, na=True)
    df = df[mask]
    df['countries_en']= 'France'  

    return df
