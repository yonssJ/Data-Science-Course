
from dash import  dcc, html
import dash_bootstrap_components as dbc
from components.data_read import Y_test, options
from dash.dependencies import Input, Output
import dash_daq as daq
import dash_bootstrap_components as dbc
from components.Figures import my_waterfall
from components.Figures import my_decision_plot
from components.Figures import my_waterfall_sum
from components.Figures import my_shap_summary
from components.Figures import bivarie_analise
from components.Figures import univarie_analise
from components.Figures import plot_scatter_projection
from components.data_read import shap_values_df, explainer,X_sample,X_samples,df1,Y_pred,dataperso
from components.data_read import X_train_sample,Y_train_sample
from components.data_read import selected_model
from sklearn.neighbors import NearestNeighbors
#import plotly.express as px

import pandas as pd
import numpy as np

import plotly.graph_objects as go

def ini_layout(app):
    app.layout = html.Div([ 
        html.Div([
            html.Div([
                html.H1('Implémentez un modèle de scoring ', id='title',
                        style = {'textAlign': 'center','color': '#FFFFFF',
                                 'fontSize': '28px','padding-top': '18px'},
                        ),
                ],
                style = {'backgroundColor': '#1f3b4d','height': '80px',
                     'display': 'flex', 'flexDirection': 'column','justifyContent': 'center'},
            ),
            dbc.Row([ 
                dbc.Col([ 
                    dcc.Location(id='url', refresh=False),
                    html.Div(id='page-content'), 
                    html.Div([
                        ],
                        style={"width": "100"},),
                    dbc.Col([
                        dbc.Col([
                            dbc.CardBody(id = 'forecast-container'),
                            ], id='forecast-container2'),
                        dbc.Col([
                            dbc.CardBody(id = 'forecast-container_check'),
                            ], id = 'forecast-container_check_1'),
                    ]),
                    ],style = {"width": "30%",'display': 'inline-block','padding-left': '100px','padding-top': '20px', 'color':'dimgrey','font-weight': 'bold'}),
                dbc.Col([ 
                    dbc.Card([html.Div(id = 'gauge')],style = {'display': 'inline-block','padding-top': '20px', "border": "none", 'color':'dimgrey','font-weight': 'bold'}),
                    dbc.Card([html.Div(id='gauge2')],style = {'display': 'inline-block','padding-top': '20px', "border": "none", 'color':'dimgrey','font-weight': 'bold'}),
                ],),
                dbc.Row([ 
                    dbc.Col([
                        html.Div(id = 'forecast-container3'),], id='forecast-container4'),
                ],style = {"width": "40%",'display': 'inline-block','padding-top': '20px', 'color':'dimgrey', 'border': '0px'}),
            ]),
            dbc.Row([ 
                dbc.Col([ 
                    html.Div(id = 'my_graph'),],
                    style = { 'display': 'inline-block','padding-left': '100px','padding-top': '10px'}
                    ),
                dbc.Col([
                    html.Div(id = 'my_graph2'),],
                    style = {'width': '25%','display': 'inline-block','padding-right': '100px','padding-top': '10px'}
                    ),
            ]),
            dbc.Row([ 
                dbc.Col([ 
                    html.Div(id = 'sumplot'),],
                    style = { 'display': 'inline-block','padding-left': '100px','padding-top': '10px'}
                    ),
                dbc.Col([
                    html.Div(id = 'sumplot2'),],
                    style = {'width': '25%','display': 'inline-block','padding-right': '100px','padding-top': '10px'}
                    ),
                ]),
            ]),

            dbc.Row([ 
                dbc.Col([ 
                    html.Div([
                        dcc.Dropdown(
                            df1.columns,
                            "CNT_CHILDREN",
                            id='varcolumn'
                        ),], style={'width': '45%','display': 'inline-block','padding-top': '30px'}),

                    html.Div(id='indicator-graphic2',style={'margin-top': -5}),
                ],style = {'display': 'inline-block','padding-left': '100px'}),
                dbc.Col([ 
                    html.Div([
                        dcc.Dropdown(
                            df1.columns,
                            "CNT_CHILDREN",
                            id='xaxis-column'
                        ),], style={'width': '45%','display': 'inline-block','padding-top': '30px'}),
                    html.Div([
                        dcc.Dropdown(
                            df1.columns,
                            "CODE_GENDER_0",
                            id='yaxis-column'
                        ),], style={'width': '45%','display': 'inline-block','padding-top': '30px'}),

                    html.Div(id='indicator-graphic',style={'margin-top': -5}),
                ],style = {'display': 'inline-block','padding-right': '100px'})
            ]),
        ])
    

    @app.callback(
        Output('forecast-container', 'children'),
        Output('forecast-container2','style'),
        Output('forecast-container3', 'children'),
        Output('forecast-container4', 'style'),
        Output('forecast-container_check','children'),
        Output('forecast-container_check_1','style'),
        Output('my_graph','children'),
        Output('my_graph2','children'),
        Output('gauge','children'),
        Output('gauge2','children'),
        Output('sumplot','children'),
        Output('sumplot2','children'),
        Output('indicator-graphic', 'children'),
        Output('indicator-graphic2', 'children'),
              [Input('url', 'pathname')],
              [Input('xaxis-column', 'value')],
              [Input('yaxis-column', 'value')],
              [Input('varcolumn', 'value')],
               )
    def update_decision(pathname,xaxis_column_name, yaxis_column_name,yvar_column): 
        name = int(pathname.split('/')[-1])
        dff_pred=df1.loc[name,selected_model]

        if dff_pred==1 :
            fig= "Client à risque -- Refus "
            styles ={'padding-top': '3px','fontSize': '26px', 'color':'red'}
        else:
            fig= "Client en regle --  Accord "
            styles ={'padding-top': '3px','fontSize': '26px', 'color':'blue'}
       
        individue=X_samples[X_samples['SK_ID_CURR']==name].index[0] 
        max_display=10
        why= "Contribution à la décision"
        fig2= my_waterfall(shap_values_df,explainer.expected_value, individue, max_display,why)
        fig2=html.Div(dcc.Graph(id='bar chart',figure=fig2))
    #
        why2="Contribution à la prédiction" 
        [fig3, cumsum]= my_decision_plot(shap_values_df,explainer.expected_value,individue, max_display, why2)
        if cumsum[max_display]>0.5 and dff_pred==0:
            fig_check= "ATTENTION: Veuillez vérifier la décision", html.Br(), "Inconsistance entre la valeur predite et le cheminement de la prédiction"
            styles_check ={"margin-top": -40,"border": "none",'fontSize': '18px', 'color':'red'}
        elif cumsum[max_display]<0.5 and dff_pred==1:
            fig_check= "ATTENTION: Veuillez vérifier la décision", html.Br(), "Inconsistance entre la valeur predite et le cheminement de la prédiction"
            styles_check ={"margin-top": -40,"border": "none",'fontSize': '18px', 'color':'red'}
        else:
            fig_check = None
            styles_check ={}
        fig3=html.Div(dcc.Graph(id='bar chart2',figure=fig3))
    #
        gauge=html.Div(
            daq.Gauge(
            id='gauges',
            color={"gradient":True,"ranges":{"blue":[0,0.4],"darkmagenta": [0.4,0.6],"red":[0.6,1]}},
            value=dff_pred,
            label='Prédiction',
            max=Y_pred.max(),
            min=Y_pred.min(),size=120,
            )
        )
    #
        gauge2=html.Div(
            daq.Gauge(
                id='gauges2',
                color={"gradient":True,"ranges":{"blue":[0,0.4],"darkmagenta": [0.4,0.6],"red":[0.6,1]}},
                value=cumsum[max_display],
                label='Chemin. prédiction',
                max=Y_pred.max(),
                min=Y_pred.min(),
                size=120,
                )   
            )
    #
        
        table_header = [
         html.Thead(html.Tr([html.Th("Information personnelle"), html.Th("  ")]))
        ]
        selected_client=name
        row1 = html.Tr([html.Td("Genre: "), html.Td(dataperso.loc[selected_client,'CODE_GENDER'])])
        row2 = html.Tr([html.Td("Etat civil: "), html.Td(dataperso.loc[selected_client,'NAME_FAMILY_STATUS'])])
        row3 = html.Tr([html.Td("Occupation:"), html.Td(dataperso.loc[selected_client,'NAME_INCOME_TYPE'])])

        table_body = [html.Tbody([row1, row2, row3])]

        styles_perso={'padding-top': '20px','fontSize': '14px', 'color':'dimgrey','padding-right': '100px'}
        perso= html.Div([dbc.Table(table_header + table_body, bordered=False,id="table-color",size='sm')])
        
        title="Contribution à la prédiction moyenne"
        fig4=my_shap_summary(shap_values_df,X_sample,max_display, title)
        
        from sklearn.neighbors import NearestNeighbors
        knn=NearestNeighbors(n_neighbors=20)  
        #On entraîne le modèle :

        random_samp = X_train_sample.sample(1000).index
        X_train_sample1=X_train_sample.loc[random_samp]
        Y_train_sample1=Y_train_sample.loc[random_samp]

        knn.fit(X_train_sample1)
        idx = knn.kneighbors(X=X_sample.loc[individue: individue],
                       n_neighbors=500,
                       return_distance=False).ravel()

        nearest_cust_idx = list(X_train_sample1.iloc[idx].index)
        print(name)
        fig4=plot_scatter_projection(X_train_sample1,
                        Y_train_sample1,#.replace({0: 'Accord', 1: 'Refus'}),
                        X_train_sample1.loc[nearest_cust_idx],
                        X_samples.loc[individue: individue],
                        name,dff_pred,
                        X_train_sample1.columns[1:3])


        fig4=html.Div(dcc.Graph(id='bar chart4',figure=fig4))


        title="Contribution à la prédiction moyenne"
        fig5=my_waterfall_sum(shap_values_df, X_sample, max_display,title)
        fig5=html.Div(dcc.Graph(id='bar chart3',figure=fig5))

        figa=bivarie_analise(X_samples,xaxis_column_name,yaxis_column_name,name)
        figa=html.Div(dcc.Graph(id='bara',figure=figa))

        figb=univarie_analise(X_samples,yvar_column,name)
        figb=html.Div(dcc.Graph(id='baraa',figure=figb))

        return fig, styles, perso,styles_perso, fig_check, styles_check , fig2, fig3, gauge, gauge2, fig4 , fig5,figa,figb

