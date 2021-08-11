import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from  pathlib import Path
from matplotlib.patches import Rectangle
from scipy import stats
import re



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

def plot_box_plot(df,xlabel_name,ylabel_name,figure_name):
    color = dict(boxes='black', whiskers='black', medians='red', caps='black')
    ax = df.plot.box(color=color,whiskerprops = dict(linestyle='-',linewidth=2.0, color='black'),figsize=(24, 8),fontsize=18)
    ax.set_xlabel(xlabel_name,fontsize=18)
    ax.set_ylabel(ylabel_name,fontsize=18)
    plt.grid(color='k', linestyle='-', linewidth=.1)
    plt.xticks(rotation=90)
    ax.label_outer() 
    plt.savefig(figure_name,format='jpeg',dpi=100,bbox_inches='tight')

def read_csv_morceaux_nutriscore(file1,maxrow,deltaline,nutriscore,directory,globalValue):
    df_in = pd.read_csv(file1,sep="\t", encoding="utf-8",skiprows=range(1,maxrow), nrows=deltaline, low_memory=False)
    # Pour enlever les NaN que sont dans 'countries_en' e remplacer par  np.nan
    df_in = df_in.replace('NAN', np.nan) 
    mask = df_in[nutriscore].notna()
    df_in = df_in[mask]
    New_file = Path(directory +"/" + str(globalValue) + str(maxrow+deltaline)+".csv")
    df_in.to_csv (New_file,  sep='\t', encoding='utf-8') 
    print ("Creer le fichier pour les donnees avec nutriscore fournit (different de NaN)", maxrow + deltaline)

def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i+1

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
    from sklearn import linear_model
    regr = linear_model.LinearRegression()
    
    X = df[xvar]
    y = df[yvar]
    regr.fit(X, y)
    regr.predict(df)

def percentage_error(df,atual,pred,value_score,group):
    if group=="a,b,c" :
        mask = (df[atual] < (value_score))
        maskb = (df[pred] < (value_score))
    else:
        mask = (df[atual] > (value_score))
        maskb = (df[pred] > (value_score))
        
    commun_products=np.sum(mask & maskb)
    Vmask_actual=df["Actual"][mask].count()
    Vmask_Predict=df["Predicted"][maskb].count()
    difference = 100*(1-commun_products/Vmask_actual)
    mask_len=len(mask)
    
    print('For nutriscore',group)
    print('Groups  ', 'Actual  ', "Predicted  ", "commun_products ", "% difference       ", "Length mask")
    print(group,'    ', Vmask_actual,'       ',Vmask_Predict,'             '   ,commun_products,'      ' 
      ,'' , difference.round(1),'            ',  mask_len)

    return Vmask_actual, Vmask_Predict, commun_products, difference, mask_len

def display_circles(pcs, n_comp, pca, axis_ranks, labels=None, label_rotation=0, lims=None):
    for d1, d2 in axis_ranks: # On affiche les 3 premiers plans factoriels, donc les 6 premières composantes
        if d2 < n_comp:

            # initialisation de la figure
            fig, ax = plt.subplots(figsize=(10,8))

            # détermination des limites du graphique
            if lims is not None :
                xmin, xmax, ymin, ymax = lims
            elif pcs.shape[1] < 30 :
                xmin, xmax, ymin, ymax = -1, 1, -1, 1
            else :
                xmin, xmax, ymin, ymax = min(pcs[d1,:]), max(pcs[d1,:]), min(pcs[d2,:]), max(pcs[d2,:])

            # affichage des flèches
            # s'il y a plus de 30 flèches, on n'affiche pas le triangle à leur extrémité
            if pcs.shape[1] < 30 :
                plt.quiver(np.zeros(pcs.shape[1]), np.zeros(pcs.shape[1]),
                   pcs[d1,:], pcs[d2,:], 
                   angles='xy', scale_units='xy', scale=1, color="grey")
                # (voir la doc : https://matplotlib.org/api/_as_gen/matplotlib.pyplot.quiver.html)
            else:
                lines = [[[0,0],[x,y]] for x,y in pcs[[d1,d2]].T]
                ax[0].add_collection(LineCollection(lines, axes=ax[0], alpha=.1, color='black'))
                  
            # affichage des noms des variables  
            if labels is not None:  
                for i,(x, y) in enumerate(pcs[[d1,d2]].T):
                    if x >= xmin and x <= xmax and y >= ymin and y <= ymax :
                        plt.text(x, y, labels[i], fontsize='14', ha='center', va='center', rotation=label_rotation, color="blue", alpha=0.5)
            
            # affichage du cercle
            circle = plt.Circle((0,0), 1, facecolor='none', edgecolor='b')
            plt.gca().add_artist(circle)

            # définition des limites du graphique
            plt.xlim(xmin, xmax)
            plt.ylim(ymin, ymax)
        
            # affichage des lignes horizontales et verticales
            plt.plot([-1, 1], [0, 0], color='grey', ls='--')
            plt.plot([0, 0], [-1, 1], color='grey', ls='--')

            # nom des axes, avec le pourcentage d'inertie expliqué
            plt.xlabel('F{} ({}%)'.format(d1+1, round(100*pca.explained_variance_ratio_[d1],1)))
            plt.ylabel('F{} ({}%)'.format(d2+1, round(100*pca.explained_variance_ratio_[d2],1)))

            plt.title("Cercle des corrélations (F{} et F{})".format(d1+1, d2+1))
            #plt.show(block=False)
            
def scatter_plot_nutriscore(x_pca,y_pca,y,cmapy,normy,xaxis,yaxis,xlabel_n,ylabel_n,clabel_n,bounds,nutriscore):
    plt.figure(figsize=(10,8))
    plt.axvline(0, ls='--') 
    plt.axhline(0, ls='--')
    ax=plt.scatter(x_pca,y_pca,c=y,cmap=cmapy,norm=normy) 
    plt.xlim(xaxis )
    plt.ylim(yaxis )
    plt.xlabel(xlabel_n)
    plt.ylabel(ylabel_n)
    cbar=plt.colorbar(ax,label=clabel_n)
    tick_locs = ([bounds[t]+(bounds[t-1]-bounds[t])/2 for t in range(1,len(bounds)) ])
    cbar.set_ticks(tick_locs)
    cbar.set_ticklabels(nutriscore) 
              
def display_scree_plot(pca):
    scree = pca.explained_variance_ratio_*100
    plt.bar(np.arange(len(scree))+1, scree)
    plt.plot(np.arange(len(scree))+1, scree.cumsum(),c="red",marker='o')
    plt.xlabel("rang de l'axe d'inertie")
    plt.ylabel("pourcentage d'inertie")
    plt.title("Eboulis des valeurs propres")
    #plt.show(block=False)
    
def separete_liquid_solid(df_score,da,dss):
    liquid = df_score.loc[df_score['categories'].str.count('boissons',flags=re.IGNORECASE)>da ]
    liquid = liquid.append(df_score.loc[df_score['categories'].str.count('soupe',flags=re.IGNORECASE)>dss ] ) 
    liquid = liquid.append(df_score.loc[df_score['categories'].str.count('Sauces',flags=re.IGNORECASE)>da ] ) 
    liquid = liquid.append(df_score.loc[df_score['categories'].str.count('Desserts',flags=re.IGNORECASE)>da ] ) 
    liquid = liquid.append(df_score.loc[df_score['product_name'].str.count("compote",flags=re.IGNORECASE)>0])

    solid = df_score[df_score['product_name'].isin(liquid['product_name']) == False]
    return liquid , solid
   
def histograme_plot(df, varname,valuevar,place,tcolors):
    g=sns.histplot(data=df[df[varname]==valuevar],color=tcolors,kde=True,ax=place)
    g.set( ylabel = valuevar)
    
def boxplot_plot(df, varname,xx,yy,valuevar,place,linewidthy,xlabell,miny,maxy,melt):
    if melt==1:
        gfg = sns.boxplot(x=xx, y=yy, data=df[df[varname]==valuevar],ax=place,linewidth=linewidthy)
        gfg.set(xlabel = xlabell, ylabel = valuevar)
    else:
        gfg = sns.boxplot(x=xx, y=yy, data=df,ax=place,linewidth=linewidthy)
        gfg.set(xlabel = xlabell, ylabel = valuevar)
        gfg.set(ylim=(miny, maxy))
        
