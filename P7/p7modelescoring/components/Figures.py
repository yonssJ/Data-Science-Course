from __future__ import annotations
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import re

backgcolor='snow'

def format_value(s, format_str):
    """ Strips trailing zeros and uses a unicode minus sign.
    """
    if not issubclass(type(s), str):
        s = format_str % s
    s = re.sub(r'\.?0+$', '', s)
    if s[0] == "-":
        s = u"\u2212" + s[1:]
    return s

def my_waterfall(shap_values_df,explaner_df,individue, max_display, why):
    base_values=explaner_df
    values = shap_values_df.iloc[individue,:]
    topx=values.abs()
    topx=topx.nlargest(max_display)  
    topx=topx.sort_values(ascending=True)
    feature_names=topx.index.to_list()
    value=values[feature_names]
    leftvalues=values.drop(feature_names)
    d=pd.Series(data={"SUM_OF_REMAIN_VARIABLES":leftvalues.sum()})
    value=pd.concat([d, value])
    topx=value.index
    valueformated=[]
    base_valuess=[]
    for i in range(len(value)):
      if i==0:
        base_valuess.append(value[i]+base_values)
      else:
        base_valuess.append(base_valuess[i-1] + value[i])
    base_valuess=list(base_valuess )
    for i in range(len(value)):
        valueformated.append(format_value(value[i], '%+0.02f'))
    
    fig2=go.Figure()
    fig2.add_trace(go.Bar( orientation = "h",
        x=value,
        y=topx,
        base=base_valuess,
        marker_color=['red' if value[i]>0 else 'blue' for i in range(len(value))],
        text=valueformated,
        textfont_size=16
    ))

    #fig2.add_vline(x=base_values, line_width=3, line_dash="solid", line_color="black", opacity=0.2)
    fig2.update_layout(
        title = '<b>'+why+'</b>', title_font_size = 18,
        showlegend = False,
        height=100+len(topx)*40,
        plot_bgcolor = backgcolor,
        margin=go.layout.Margin(
                l=5,
                r=5,
                b=40,
                t=70,
                pad=4
        ),
        font=dict(color='dimgrey'),
    )
    return fig2
  
def my_decision_plot(shap_values_df,explaner_df,individue, max_display, why):
    base_values=explaner_df
    values = shap_values_df.iloc[individue,:]
    topx=values.abs()
    topx=topx.nlargest(max_display)  
    topx=topx.sort_values(ascending=True)
    feature_names=topx.index.to_list()
    value=values[feature_names]
    leftvalues=values.drop(feature_names)
    d=pd.Series(data={"REMAIN_VARIABLES":leftvalues.sum()})
    value=pd.concat([d, value])
    topx=value.index
    valueformated=[]
    for i in range(len(value)):
        valueformated.append(format_value(value[i], '%+0.02f'))
    
    base_valuess=[]
    for i in range(len(value)):
      if i==0:
        base_valuess.append(base_values+value[i])
      else:
        base_valuess.append(base_valuess[i-1]+ value[i])
    
    base_valuess=np.array(list(base_valuess))
    base_valuess=1/(1+np.exp(-base_valuess))         # metre en log scale np.log(-cvalue/(1-cvalue)) #

    # Create traces
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=base_valuess, y=topx,
        mode='markers',
        marker=dict(
        size=16,
        color=base_valuess,          
        colorscale="Bluered", 
        showscale=True,
        colorbar=dict(thickness=10, tickvals=[min(base_valuess), max(base_valuess)], ticktext=['Faible', 'Elevé'])  )
    ))
    fig3.add_vline(x=base_valuess[0], line_width=3, line_dash="dash", line_color="gray", opacity=0.2)
    fig3.update_layout(
        title='<b>'+why+'</b>',
        title_font_size = 18,
        showlegend = False,
        height=100+len(topx)*40,
        plot_bgcolor = backgcolor,
        margin=go.layout.Margin(
                l=5,
                r=5,
                b=40,
                t=70,
                pad=4
        ),
        font=dict(color='dimgrey'),
    )
    return fig3, base_valuess

def _trim_crange(values, nan_mask):
    """Trim the color range, but prevent the color range from collapsing."""
    # Get vmin and vmax as 5. and 95. percentiles
    vmin = np.nanpercentile(values, 5)
    vmax = np.nanpercentile(values, 95)
    if vmin == vmax:  # if percentile range is equal, take 1./99. perc.
        vmin = np.nanpercentile(values, 1)
        vmax = np.nanpercentile(values, 99)
        if vmin == vmax:  # if still equal, use min/max
            vmin = np.min(values)
            vmax = np.max(values)

    if vmin > vmax: # fixes rare numerical precision issues
        vmin = vmax

    # Get color values depnding on value range
    cvals = values[np.invert(nan_mask)].astype(np.float64)
    cvals_imp = cvals.copy()
    cvals_imp[np.isnan(cvals)] = (vmin + vmax) / 2.0
    cvals[cvals_imp > vmax] = vmax
    cvals[cvals_imp < vmin] = vmin

    return vmin, vmax, cvals
    
def my_waterfall_sum(shap_values_df,X_sample, max_display, why):
    value=abs(shap_values_df)
    value=value.sum()
    feature_order=value.sort_values(ascending=True)
    feature_order=feature_order.nlargest(max_display) 
    feature_names=feature_order.index.to_list() 
    na_fill=-999
    features = feature_order.values

    min_shap = shap_values_df.min().min()
    max_shap = shap_values_df.max().max()
    shap_range = max_shap - min_shap
    min_shap = min_shap - 0.01 * shap_range
    max_shap = max_shap + 0.01 * shap_range
    colorbar_trace  = go.Scatter(x=[None],
                             y=[None],
                             mode='markers',
                             marker=dict(
                                 colorscale='Bluered', 
                                 showscale=True,
                                 cmin=min_shap,
                                 cmax=max_shap,
                                 colorbar=dict(thickness=10, tickvals=[min_shap, max_shap],  ticktext=['Faible', 'Elevé'], outlinewidth=0)
                             ),
                             hoverinfo='none'
                           )
    from plotly.subplots import make_subplots
    fig =  make_subplots(rows=len(feature_order), cols=1, shared_xaxes=True)
    row_height = 0.4
    for i, col in enumerate(feature_names):
        shaps=shap_values_df[col]
        values = None if features is None else feature_order[col]
        N = len(shaps)
        nbins = 100
        quant = np.round(nbins * (shaps - np.min(shaps)) / (np.max(shaps) - np.min(shaps) + 1e-8))
        inds = np.argsort(quant + np.random.randn(N) * 1e-6)
        layer = 0
        last_bin = -1
        ys = np.zeros(N)
        for ind in inds:
            if quant[ind] != last_bin:
                layer = 0
                ys[ind] = np.ceil(layer / 2) * ((layer % 2) * 2 - 1)
                layer += 1
                last_bin = quant[ind]
            ys *= 0.9 * (row_height / np.max(ys + 1))

        nan_mask = np.isnan(values)
        # Trim the value and color range to percentiles
        vmin, vmax, cvals = _trim_crange(values, nan_mask)
        fig.add_trace(go.Scatter(
            x=shaps,
            y=i + ys,
            mode='markers',
            marker=dict(
                size=10,
                cmin=min_shap,
                cmax=max_shap ,
                color=X_sample[col].replace({na_fill:np.nan}),
                colorscale='Bluered',
                showscale=False,
                opacity=0.3,
                colorbar=dict(thickness=10, tickvals=[min_shap, max_shap]),
                ),
            name=col,
            showlegend=False,
            opacity=0.8),
        row=i+1, col=1)  
    
        fig.update_xaxes(showgrid=False, zeroline=True, 
                         range=[min_shap, max_shap], row=i+1, col=1) 
        fig.update_yaxes(showgrid=False, zeroline=False,
            tickvals = [i],ticktext = [feature_names[i]], row=i+1, col=1)
        
    fig.add_vline(x=0, line_width=2, line_dash="solid", line_color="gray")
    fig.add_trace(colorbar_trace)
    fig.update_layout(
        title = '<b>'+why+'</b>', title_font_size = 18,
        showlegend = False,
        height=100+len(feature_names)*45,
        plot_bgcolor = backgcolor,
        margin=go.layout.Margin(
                l=5,
                r=5,
                b=40,
                t=70,
                pad=4
        ),
        font=dict(color='dimgrey'),   )                
    
        


    return fig

def my_shap_summary(shap_values_df,X_sample, max_display, why):
    values = shap_values_df.sum()
    topx=values.abs()
    topx=topx.nlargest(max_display)  
    topx=topx.sort_values(ascending=True)
    feature_names=topx.index.to_list()

    na_fill=-999
    min_shap = shap_values_df.min().min()
    max_shap = shap_values_df.max().max()
    shap_range = max_shap - min_shap
    min_shap = min_shap - 0.01 * shap_range
    max_shap = max_shap + 0.01 * shap_range

    colorbar_trace  = go.Scatter(x=[None],
                             y=[None],
                             mode='markers',
                             marker=dict(
                                 colorscale='Bluered', 
                                 showscale=True,
                                 cmin=min_shap,
                                 cmax=max_shap,
                                 colorbar=dict(thickness=10, tickvals=[min_shap, max_shap],  ticktext=['Faible', 'Elevé'], outlinewidth=0)
                             ),
                             hoverinfo='none'
                           )
    from plotly.subplots import make_subplots
    fig =  make_subplots(rows=len(feature_names), cols=1, shared_xaxes=True)
    for i, col in enumerate(feature_names): #display_columns):
        fig.add_trace(go.Scatter(
            x=shap_values_df[col],
            y=np.random.rand(len(X_sample)),
            mode='markers',
            marker=dict(
                size=5,
                cmin=min(shap_values_df[col]),
                cmax=max(shap_values_df[col]),
                color=X_sample[col].replace({na_fill:np.nan}),
                colorscale='Bluered',
                showscale=False,
                opacity=0.3,
                colorbar=dict(thickness=10, tickvals=[min(shap_values_df[col]), max(shap_values_df[col])]),
                ),
            name=col,
            showlegend=False,
            opacity=0.8),
        row=i+1, col=1) 
    
        fig.update_xaxes(showgrid=False, zeroline=True, 
                         range=[-2, 2], row=i+1, col=1) #min_shap, max_shap
        fig.update_yaxes(showgrid=False, zeroline=False,
                         showticklabels=False, row=i+1, col=1)
    fig.add_vline(x=0, line_width=2, line_dash="solid", line_color="gray")
    fig.add_trace(colorbar_trace)
    fig.update_layout(
        title = '<b>'+why+'</b>', title_font_size = 18,
        showlegend = False,
        height=100+len(feature_names)*50,
        plot_bgcolor = backgcolor,
        margin=go.layout.Margin(
                l=5,
                r=5,
                b=40,
                t=70,
                pad=4
        ),
        font=dict(color='dimgrey'),   )                 
    
    return fig
def bivarie_analise(df1,xaxis_column_name,yaxis_column_name,name):
    df2=df1[df1['SK_ID_CURR']==name]
    figa = go.Figure()
    figa.add_trace(go.Scatter(x=df1[xaxis_column_name],
                     y=df1[yaxis_column_name],name="Autres",mode='markers')
            )
    figa.add_trace(go.Scatter(x=df2[xaxis_column_name],
                     y=df2[yaxis_column_name],name="Client "+str(name),
                     marker=dict(size=8,
                              line=dict(width=2,
                                        color='red')),
                     mode='markers'
                     
                     )
            )
    figa.update_xaxes(title=xaxis_column_name)

    figa.update_yaxes(title=yaxis_column_name)
    figa.update_layout(
        plot_bgcolor = backgcolor,
        margin=go.layout.Margin(
                l=5,
                r=5,
                b=40,
                t=70,
                pad=4
        ),
        font=dict(color='dimgrey'),   ) 

    return figa
def univarie_analise(df1,xvarcolumn_name,name):
    df2=df1[df1['SK_ID_CURR']==name]
    fig = go.Figure()
    fig.add_trace(go.Histogram(x = df1[xvarcolumn_name])) #,name="Autres"))
    fig.update_layout(xaxis_title=xvarcolumn_name, yaxis_title="Count")
    fig.add_trace(go.Scatter(x = df2[xvarcolumn_name], 
    name="Client "+str(name),
    
                     marker=dict(size=8,
                              line=dict(width=5,
                                        color='red')),
                     mode='markers'
                     ))
    fig.update_layout(barmode='stack') 
    fig.update_layout(
        plot_bgcolor = backgcolor,
        margin=go.layout.Margin(
                l=5,
                r=5,
                b=40,
                t=70,
                pad=4
        ),
        font=dict(color='dimgrey'),   )  
        
    return fig
    
def plot_scatter_projection(X, ser_clust, plot_highlight, X_cust,name_customer,dff_pred, columns):
    X_all = pd.concat([X, X_cust], axis=0)
    ind_neigh = list(plot_highlight.index)
    df_data = X_all.loc[:, columns]
    
    customer_idx = X_cust.index[0]
    value=ser_clust["TARGET"]
    value=value[ind_neigh]
    
    marker_color_0=[]
    marker_color_1=[]
    x_1=[]
    y_1=[]
    x_0=[]
    y_0=[]
    x=df_data.loc[ind_neigh].iloc[:, 0]
    y=df_data.loc[ind_neigh].iloc[:, 1]
    for i in value.index:
        if value[i]>0:
            cx='red'
            marker_color_1.append(cx)
            x_1.append(x[i])
            y_1.append(y[i])
            name_1="Refus"
        else: 
            cx='blue'
            marker_color_0.append(cx)
            x_0.append(x[i])
            y_0.append(y[i])
            name_0="Accord"
    fig = go.Figure()
    fig.add_trace(
            go.Scatter(
            x=x_0,
            y=y_0,
            name=str(name_0),
            mode='markers',
            marker=dict(
                size=10,color=marker_color_0)
        )
    )

    fig.add_trace(
            go.Scatter(
            x=x_1,
            y=y_1,
            name=str(name_1),
            mode='markers',
            marker=dict(
                size=10,color=marker_color_1)

        )
    )
    if dff_pred==1:
        ccolor_cst='red'
    else:
        ccolor_cst='blue'
    fig.add_trace(go.Scatter(x=df_data.loc[customer_idx:customer_idx].iloc[:, 0],
                y=df_data.loc[customer_idx:customer_idx].iloc[:, 1],
                name="Client " + str(name_customer),
                mode='markers',
                marker=dict(
                    size=20,
                    color=ccolor_cst,
                    line=dict(
                        color='Black',
                        width=4))
            ))
    
    fig.update_xaxes(title='CNT_CHILDREN')

    fig.update_yaxes(title='AMT_INCOME_TOTAL')
    fig.update_layout(
        title = '<b>'+"Clients avec des profiles similaires"+'</b>', title_font_size = 18,
        showlegend = True,
        height=180*3,
        plot_bgcolor = backgcolor,
        margin=go.layout.Margin(
                l=5,
                r=5,
                b=40,
                t=70,
                pad=4
        ),
        font=dict(color='dimgrey'))

    return fig
