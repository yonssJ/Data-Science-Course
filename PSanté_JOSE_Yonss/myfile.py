import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from  pathlib import Path
from matplotlib.patches import Rectangle
from scipy import stats


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

def isnull_values_sum(df,define_max_acepteble_nan):
    taille = len(df)/define_max_acepteble_nan
    print(taille)
    null   = df.isnull().sum() 
    b=null.le(taille)
    cols_to_keep = df.columns[b]
    df = df.loc[:, cols_to_keep]
    return df
def variables_with_nutriscore(df,nutriscore_values):
    taille = nutriscore_values
    print('Nutriscore variables with Nan ', intaille-1)
    null   = df.isnull().sum() 
    b=null.le(taille)
    cols_to_keep = df.columns[b]
    df = df.loc[:, cols_to_keep]
    return df

def plot_box_plot(df,xlabel_name,ylabel_name,figure_name):
    color = dict(boxes='black', whiskers='black', medians='red', caps='black')
    ax = df.plot.box(color=color,whiskerprops = dict(linestyle='-',linewidth=2.0, color='black'),figsize=(24, 8),fontsize=18)
    ax.set_xlabel(xlabel_name,fontsize=18)
    ax.set_ylabel(ylabel_name,fontsize=18)
    plt.grid(color='k', linestyle='-', linewidth=.1)
    plt.xticks(rotation=90)
    ax.label_outer() 
    plt.savefig(figure_name,format='jpeg',dpi=100,bbox_inches='tight')

def read_csv_morceaux(file1,maxrow,deltaline,country,rpays,country_interests,directory,globalValue):
    df_in = pd.read_csv(file1,sep="\t", encoding="utf-8",skiprows=range(1,maxrow), nrows=deltaline, low_memory=False)
    # Pour enlever les NaN que sont dans 'countries_en' e remplacer par  np.nan
    df_in = df_in.replace('NAN', np.nan) 
    # Selectioner seulement les donnees du marche Français vu que le projet est pour la France
    mask = df_in[country].str.contains(rpays, na=True)
    df_in = df_in[mask]
    df_in[country]= country_interests 
    # Creer un fichier .csv avec les donnees du marché Francais
    New_file = Path(directory +"/" + str(globalValue) + str(maxrow+deltaline)+".csv")
    df_in.to_csv (New_file,  sep='\t', encoding='utf-8') 
    print ("Creer le fichier pour le marché Français", maxrow + deltaline)

def read_csv_morceaux_nutriscore(file1,maxrow,deltaline,nutriscore,directory,globalValue):
    df_in = pd.read_csv(file1,sep="\t", encoding="utf-8",skiprows=range(1,maxrow), nrows=deltaline, low_memory=False)
    # Pour enlever les NaN que sont dans 'countries_en' e remplacer par  np.nan
    df_in = df_in.replace('NAN', np.nan) 
    mask = df_in[nutriscore].notna()
    df_in = df_in[mask]
    New_file = Path(directory +"/" + str(globalValue) + str(maxrow+deltaline)+".csv")
    df_in.to_csv (New_file,  sep='\t', encoding='utf-8') 
    print ("Creer le fichier pour le donnees avec nutriscore fournit (different de NaN)", maxrow + deltaline)

def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i+1

def group_and_loc_group(df,categ):
    grouped = df.groupby(categ, as_index=False)
    df_grouped=[]
    for i in grouped.groups.keys():
        loc=grouped.groups[i]
        df_grouped.append(pd.DataFrame(df.iloc[loc]))
    return grouped, df_grouped

def plot_function(x, y, plot_type,figure_name,xlabel_name,ylabel_name,plot_legend):
    fig, ax   = plt.subplots(figsize=(24, 8))

    ax.plot(x, y, marker = plot_type)
    if len(plot_legend)<6:
           ncol = len(plot_legend)
    else:
           ncol =int(len(plot_legend)/2)
           
    plt.legend(plot_legend, ncol =ncol, loc='upper center',bbox_to_anchor=(0.5, 1.2))
    ax.set_xlabel(xlabel_name,fontsize=18)
    ax.set_ylabel(ylabel_name,fontsize=18)
    plt.rcParams['font.size'] = '18'
    plt.grid(color='k', linestyle='-', linewidth=.1)
    fig.savefig(figure_name,format='jpeg',dpi=100,bbox_inches='tight')

def correl_pvalue(df)  :
    coeffmat= np.zeros((df.shape[1], df.shape[1]))
    pvalmat = np.zeros((df.shape[1], df.shape[1]))
    for i in range(df.shape[1]):
        for j in range(df.shape[1]):        
            corrtest = stats.pearsonr(df[df.columns[i]], df[df.columns[j]])  

            coeffmat[i,j] = corrtest[0]
            pvalmat[i,j] = corrtest[1]

    dfcoeff = pd.DataFrame(coeffmat, columns=df.columns, index=df.columns)
    dfpvals = pd.DataFrame(pvalmat, columns=df.columns, index=df.columns)
    return dfcoeff,dfpvals

def create_régression_linéaire(df,xvar,yvar)  :
#     X = np.matrix([np.ones(df.shape[0]), df[xvar].values]).T
#     y = np.matrix(df[yvar]).T

#     # On effectue le calcul exact du paramètre theta
#     theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

#     plt.xlabel(xvar)
#     plt.ylabel(yvar)

#     plt.plot(df[xvar], df[yvar], 'ro', markersize=4)

#     # On affiche la droite entre 0 et 35
#     plt.plot([df[xvar].min(),df[xvar].max()], [theta.item(0),theta.item(0) + df['nutriscore_score'].max() * theta.item(1)], linestyle='--',       c='#000000')

#     plt.show()
    from sklearn import linear_model
    regr = linear_model.LinearRegression()
    
    X = df[xvar]
    y = df[yvar]
    regr.fit(X, y)
    regr.predict(df)
#    return theta
