import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import copy
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn import decomposition, preprocessing
from pathlib import Path
from yellowbrick.cluster import KElbowVisualizer
from yellowbrick.cluster import InterclusterDistance
#from bioinfokit.visuz import cluster
from yellowbrick.cluster import intercluster_distance


def find_optimal_K(rstate,x_norm):
    #
    visualizer = KElbowVisualizer(KMeans(random_state=rstate))
    visualizer.fit(x_norm)
    visualizer.show() 
    
    # Find optimal clusters number
    nclusters= visualizer.__dict__['elbow_value_']
    
    #Utilisation de l'analyse en composantes principales pour visualiser les clusters
    intercluster_distance(KMeans(nclusters), 
                      x_norm, 
                      embedding='tsne', 
                      random_state=rstate)
    return nclusters

def X_selection_Standardize_features(df,var2drop):
    df=df.select_dtypes(exclude='object')
    df=df.drop(var2drop,axis=1)
    df.plot(kind='density',sharex=True,figsize=(15,5),layout=(10,1))
    feat_cols=df.columns
    scaler = preprocessing.StandardScaler().fit(df)
    df_x_norm =  scaler.transform(df)
    df_x_norm = pd.DataFrame(data=df_x_norm,index=df.index,columns=df.columns)
    df_x_norm.plot(kind='density',sharex=True,figsize=(15,5),layout=(10,1))

    return df_x_norm, feat_cols

def final_KMeans_model(df,nclusters,nstate):
    kmeans_visualizer = KMeans(n_clusters=nclusters,random_state=nstate)
    kmeans_visualizer.fit(df)
    name='Cluster = '+str(nclusters)
    df[name]=kmeans_visualizer.predict(df)
    labels =kmeans_visualizer.labels_
    centers=kmeans_visualizer.cluster_centers_
    return df,labels,centers,name    

def plot_radar(df,fig,ax,index,i,name,palette,kx):
    labels = df[1:].columns                                    
    if ax[i]==ax[0]:
        index = df.index 
    num_vars = len(labels)                                                
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist() 
    angles += angles[:1]                                                  
   
    for knx in df.index:
        add_to_radar(df,knx, palette[knx],angles,labels,ax[i],fig,kx)
        #Add legend
        ax[i].legend(loc='upper right', bbox_to_anchor=(0.1, 0.1),labels=index)
        ax[i].set_title(name,fontsize=18,fontweight="bold")

def pca_dim_reduction(df,var_explaned,figure_name ):
    pcamodel=decomposition.PCA(var_explaned)
    pcamodel.fit(df)
    explained_var=pcamodel.explained_variance_ratio_
    explained=pcamodel.explained_variance_ratio_.cumsum()
    loadings = pcamodel.components_
    num_pc = len(pcamodel.components_) #pca.n_features_
    pc_list = ["PC"+str(i) for i in list(range(1, num_pc+1))]
#
    loadings_df = pd.DataFrame.from_dict(dict(zip(pc_list, loadings)))
    loadings_df['variable'] = df.columns.values
    loadings_df = loadings_df.set_index('variable')
    
    df_dim_red=pcamodel.transform(df)
    df_dim_red = pd.DataFrame(data=df_dim_red, columns=pc_list)

    #figure_name =Path(str(dir_fig) +'/' + str('percentagen_explique.jpeg'))

    fig, ax = plt.subplots(1,2,figsize=(12,5))
    fig.subplots_adjust(right=1.2)
#
    sns.heatmap(loadings_df,vmin=-1,vmax=1, ax=ax[0], annot=True, cmap='Spectral')
#
    display_scree_plot(pcamodel,pc_list,ax[1])

    plt.savefig(figure_name,format='jpeg',dpi=100,bbox_inches='tight')
    
    return df_dim_red,pcamodel

def stability_measure(nclusters,df):
    inx=0
    inx0=0
    inj=0
    df_stability=pd.DataFrame()
    #
    for knx in range(0,nclusters*2): 
        knx=nclusters
        kmeans_visualizer = KMeans(n_clusters=knx,init='random')
        kmeans_visualizer.fit(df)
        name1='Iteration  '+str(inj)
        df_stability[name1]=kmeans_visualizer.predict(df)
    
        inj0=inj-1
        if inj==0:
            name0=name1
            inj0=inj
        ARI_RFM_KMeans=adjusted_rand_score(df_stability[name1],df_stability[name0])
        print('ARI  '+ str(name1) + ' - ' + str(name0)+'     =  ' , ARI_RFM_KMeans ) 
        name0=name1
        inj=inj+1  
    return df_stability

def myplot(data,x_scaled,name1,pca,dim,boundary,alpha,labels=None):
    fig, ax = plt.subplots(figsize=(15,8))
    
    d1=dim[0]
    d2=dim[1]
    c=x_scaled[name1]
    coeff=np.transpose(pca.components_[d1:d2+1, :])
    score=data.values 
    xs = score[:,d1]
    ys = score[:,d2]
    n = coeff.shape[0]
    value=np.unique(c)
    scalex = 1.0/(xs.max() - xs.min())
    scaley = 1.0/(ys.max() - ys.min())
    ass=ax.scatter(xs * scalex,ys * scaley,c=c.astype(float),cmap='Paired',s=50, alpha=alpha)
    ax.set_xlim([-boundary,boundary])
    ax.set_ylim([-boundary,boundary])
    legend1 = ax.legend(*ass.legend_elements(),
                    bbox_to_anchor=(1.05, 1),loc="upper left", title=name1)
    ax.add_artist(legend1)
    for i in range(n):
        ax.arrow(0, 0, coeff[i,0], coeff[i,1],color = 'black',alpha = 0.5)
        if labels is None:
            ax.text(coeff[i,0]* 1.3, coeff[i,1] * 1.15, "Var"+str(i+1), color = 'black', ha = 'center', va = 'center')
        else:
            ax.text(coeff[i,0]* 1.3, coeff[i,1] * 1.15, labels[i], color = 'black', ha = 'center', va = 'center')
 
    ax.set_xlabel("PC{} ({}%)".format(d1+1, round(100*pca.explained_variance_ratio_[d1],1)))
    ax.set_ylabel("PC{} ({}%)".format(d2+1, round(100*pca.explained_variance_ratio_[d2],1)))
       
    ax.axvline(0,color='grey', ls='--') 
    ax.axhline(0,color='grey', ls='--')
    ax.grid()
  

# def myplot_PCA(data,c,pca,dim,boundary,alpha,labels=None):
#     fig = plt.figure(figsize=(12,8))
    
#     d1=dim[0]
#     d2=dim[1]
#     print(d1,d2)
#     coeff=np.transpose(pca.components_[d1:d2+1, :])
#     score=data.values 
#     xs = score[:,d1]
#     ys = score[:,d2]
#     n = coeff.shape[0]
#     scalex = 1.0/(xs.max() - xs.min())
#     scaley = 1.0/(ys.max() - ys.min())
#     plt.scatter(xs * scalex,ys * scaley,c=c,cmap='Paired',s=50, alpha=alpha)
#     plt.xlim([-boundary,boundary])
#     plt.ylim([-boundary,boundary])
    
#     for i in range(n):
#         plt.arrow(0, 0, coeff[i,0], coeff[i,1],color = 'r',alpha = 0.5)
#         if labels is None:
#             plt.text(coeff[i,0]* 1.3, coeff[i,1] * 1.15, "Var"+str(i+1), color = 'black', ha = 'center', va = 'center')
#         else:
#             plt.text(coeff[i,0]* 1.3, coeff[i,1] * 1.15, labels[i], color = 'black', ha = 'center', va = 'center')
 
#     plt.xlabel("PC{} ({}%)".format(d1+1, round(100*pca.explained_variance_ratio_[d1],1)))
#     plt.ylabel("PC{} ({}%)".format(d2+1, round(100*pca.explained_variance_ratio_[d2],1)))
       
#     plt.axvline(0,color='grey', ls='--') 
#     plt.axhline(0,color='grey', ls='--')
#     plt.grid()
    

def display_scree_plot(pca,pc_list,ax):
    scree = pca.explained_variance_ratio_*100
    ax.bar(pc_list, scree, alpha=0.5, align='center', label='individual explained variance')
    ax.step(pc_list,  scree.cumsum(),c="red",marker='o',where= 'mid', label='cumulative explained variance')
    ax.set_xlabel("PCs")
    ax.set_ylabel("Ratio de la variance expliqué")
    plt.legend(loc = 'best')
    
#     plt.bar(range(1,19), var_explained, alpha=0.5, align='center', label='individual explained variance')
# plt.step(range(1,19),cum_var_exp, where= 'mid', label='cumulative explained variance')
# plt.ylabel('Explained variance ratio')
# plt.xlabel('Principal components')
# plt.legend(loc = 'best')
# plt.show()
 
def nombre_client_segment(segments_counts,fig,ax,name):
    bars = ax.barh(range(len(segments_counts)),
              segments_counts,
              color='silver')
    ax.set_frame_on(False)
    ax.tick_params(left=False,
               bottom=False,
               labelbottom=False)
    ax.set_yticks(range(len(segments_counts)))
    ax.set_yticklabels(segments_counts.index)
    ax.set_title(name,fontsize=18,fontweight="bold")

    for i, bar in enumerate(bars):
        value = bar.get_width()
        if segments_counts.index[i] in ['Best', 'Loyal']:#['Best Customers', 'Loyal Customers']:
            bar.set_color('green')
        ax.text(value,
                bar.get_y() + bar.get_height()/2,
                '{:,} ({:}%)'.format(int(value),
                                   int(value*100/segments_counts.sum())),
                va='center',
                ha='left'
               )
def boxplot_plot(df, varname,xx,yy,valuevar,place,linewidthy,xlabell,miny,maxy,melt):
    if melt==1:
        gfg = sns.boxplot(x=xx, y=yy, data=df[df[varname]==valuevar],ax=place,linewidth=linewidthy,color='#ff9900')
        gfg.set(xlabel = xlabell, ylabel = valuevar)
        plt.rcParams['font.size'] = '18'
    else:
        gfg = sns.boxplot(x=xx, y=yy, data=df,ax=place,linewidth=linewidthy,color='#ff9900')
        gfg.set(xlabel = xlabell, ylabel = valuevar)
        gfg.set(ylim=(miny, maxy))
        plt.rcParams['font.size'] = '18'
        
    
def add_to_radar(df, cluster, color,angles,labels,ax,fig,kx):
    values = df.loc[cluster].tolist()
    values += values[:1]
    if kx==0:
        plt.plot(angles, values, color=color, linewidth=1, label=cluster)
        plt.fill(angles, values, color=color, alpha=0.25)
        plt.set_theta_offset(np.pi / 2)
        plt.set_theta_direction(-1)

        plt.set_thetagrids(np.degrees(angles[:-1]), labels)
        for label, angle in zip(plt.get_xticklabels(), angles):
            if angle in (0, np.pi):
                label.set_horizontalalignment('center')
            elif 0 < angle < np.pi:
                label.set_horizontalalignment('left')
            else:
                label.set_horizontalalignment('right')
        
    else:
        ax.plot(angles, values, color=color, linewidth=1, label=cluster)
        ax.fill(angles, values, color=color, alpha=0.25)
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)

        ax.set_thetagrids(np.degrees(angles[:-1]), labels)
        for label, angle in zip(ax.get_xticklabels(), angles):
            if angle in (0, np.pi):
                label.set_horizontalalignment('center')
            elif 0 < angle < np.pi:
                label.set_horizontalalignment('left')
            else:
                label.set_horizontalalignment('right')

def plot_box_plot(df,xlabel_name,ylabel_name,figure_name):
    color = dict(boxes='black', whiskers='black', medians='red', caps='black')
    ax = df.plot.box(color=color,whiskerprops = dict(linestyle='-',linewidth=2.0, color='black'),figsize=(24, 8),fontsize=18)
    ax.set_xlabel(xlabel_name,fontsize=18)
    ax.set_ylabel(ylabel_name,fontsize=18)
    plt.grid(color='k', linestyle='-', linewidth=.1)
    plt.xticks(rotation=90)
    ax.label_outer() 
    plt.savefig(figure_name,format='jpeg',dpi=100,bbox_inches='tight')
    
def histplot_plot(fig,df_numerical,poss1,possi2,feat_idx):
    for feature in df_numerical.columns: 
        if df_numerical[feature].dtype == np.int64 or df_numerical[feature].dtype == np.float64:
            feat_idx=feat_idx
            ax = fig.add_subplot(poss1,possi2, (feat_idx+1))
            fig.subplots_adjust(left=0.1,bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
            #% or count  a faire  ???????? densite true /false
            h = ax.hist(df_numerical[feature], bins=50, color='steelblue', edgecolor='none')
            ax.set_title(df_numerical.columns[feat_idx], fontsize=14)
            feat_idx=feat_idx+1
            
def plot_data_groupby(df_merge,states_selection,col_name,groupby,what_groupby,xlabel,ylabel,fig,\
                      scount,smean,ssum,axs,pos):
    for knx in col_name:
        title=str('State '+ knx)
        df1         =df_merge.loc[df_merge[states_selection]==knx]
        if scount==1:
            data_2_plot_plot = df1.groupby([groupby]).count()
        elif smean==1:
            data_2_plot_plot = df1.groupby([groupby]).mean()
        elif ssum==1:
            data_2_plot_plot = df1.groupby([groupby]).sum()
        
        Cust_charac=data_2_plot_plot.loc[:,data_2_plot_plot.columns.str.contains(what_groupby)]
        Cust_characanom=Cust_charac-Cust_charac.mean()
        
        Cust_charac.plot.bar(stacked=True,ax=axs[pos],fontsize=14).set_title(knx)
        
        pos=pos+1
    plt.ylabel(ylabel, fontsize=16)
    plt.xlabel(xlabel, fontsize=16)
