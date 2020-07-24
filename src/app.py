import os
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd


def generate_table(dataframe, max_rows=10):
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(min(len(dataframe), max_rows))
        ])
    ])


app_body_tab_overview = html.Div(
    className='app-body-container',
    children=[
        html.Div(
            className='app-body-summary-bar-top',
            children=[
                html.Div([
                    html.Span("BALANCE"),
                    html.Span("$51 245.56", style={'marginLeft': '10px', 'color': '#BB86FC'}),
                    html.Span("PROFIT", style={'marginLeft': '100px'}),
                    html.Span("$1 245.56", style={'marginLeft': '10px', 'color': '#BB86FC'}),
                    html.Span("Last update", style={'marginLeft': '100px'}),
                    html.Span("July 23, 2020", style={'marginLeft': '10px'}),
                ], style={'margin': '15px', 'float': 'left'}),
                html.Button('UPDATE', className='app-body-summary-button'),
            ]),
        html.Div(
            className='app-body-longrow-middle',
            children=[
                html.Div(["here, a plot of balance evolution (true v. theory)"], className='app-body-largecell-left'),
                html.Div(["here, a list of performances"], className='app-body-cell-right'),
            ]),
        html.Div(
            className='app-body-row-bottom',
            children=[
                html.Div(["here, a list of yesterday performances"], className='app-body-cell-left'),
                html.Div(["here, a list of predictions for next day"], className='app-body-cell-right'),
            ]),
    ])

app_body_tab_predict = [
    html.Div(
        style={'width': '32%', 'height': '96.5%', 'display': 'inline-flex', 'flexFlow': 'column',
               'padding': '10px 5px 10px 10px'},
        children=[
            html.Button("UPDATE DATA", className='app-body-update-button'),
            html.Div("✔  Data are up to date", className='app-body-status-section'),
            html.Div("Showing the N last entry in PRICES to check date", className='app-body-prices-section'),
            html.Div("Showing the N last entry in NEWS to check date", className='app-body-news-section'),
        ]
    ),
    html.Div(
        style={'width': '32%', 'height': '96.5%', 'display': 'inline-flex', 'flexFlow': 'column',
               'padding': '10px 5px 10px 5px'},
        children=[
            html.Button("COMPUTE ORDERS", className='app-body-update-button'),
            html.Div("⚠  Today's recommendation are not computed", className='app-body-status-section'),
            html.Div("Showing the N last recommendations if data is updated", className='app-body-news-section'),
        ]
    ),
    html.Div(
        style={'width': '32%', 'height': '96.5%', 'display': 'inline-flex', 'flexFlow': 'column',
               'padding': '10px 10px 10px 5px'},
        children=[
            html.Button("PLACE ORDERS", className='app-body-update-button'),
            html.Div("⚠  Orders have not been placed", className='app-body-status-section'),
            html.Div("Showing quantities if recommendation is ready", className='app-body-news-section'),
        ]
    ),
]

app_body_tab_model = html.Div(
    className='app-body-container',
    children=[
        html.Div(
            className='app-body-row-top',
            children=[
                html.Div(["Current specifications of the model: T1, T2, PCA, etc."], className='app-body-cell-left'),
                html.Div(["Training tab : Change those specs to retrain model"], className='app-body-cell-right')
            ]),
        html.Div(
            className='app-body-row-middle',
            children=[
                html.Div(["Settings and raw metrics on train"], className='app-body-cell-left'),
                html.Div(["Settings and raw metrics on val"], className='app-body-cell-middle'),
                html.Div(["Settings and raw metrics on test"], className='app-body-cell-right')
            ]),
        html.Div(
            className='app-body-summary-bar-bottom',
            children=[
                html.Div([
                    html.Span("ACCURACY"),
                    html.Span("57.25%", style={'marginLeft': '10px', 'color': '#BB86FC'}),
                    html.Span("BACKTEST MONTHLY PROFIT", style={'marginLeft': '100px'}),
                    html.Span("$10 245.56", style={'marginLeft': '10px', 'color': '#BB86FC'}),
                    html.Span("TRAINED ON", style={'marginLeft': '100px'}),
                    html.Span("July 23, 2020", style={'marginLeft': '10px'}),
                ], style={'margin': '15px', 'float': 'left'}),
                html.Button('RETRAIN', className='app-body-summary-button'),
            ]),
    ])

#############################################################################################

assets_path = os.path.join(os.getcwd(), '../resources/')
app = dash.Dash(__name__, assets_folder=assets_path)
app.title = 'Airain'

app.layout = html.Div([

    html.Div(
        id="title-bar",
        children=[html.Img(src=app.get_asset_url('airain-white.png'), className='title-bar-logo')],
    ),

    html.Div(
        id="nav-bar",
        children=[
            html.Div("NAVIGATION", className='nav-bar-header'),
            dcc.Tabs(id='tabs', value='tab-1', vertical=True,
                     parent_className='custom-tabs', className='custom-tabs-container',
                     children=[
                         dcc.Tab(label='Overview', value='tab-1', className='nav-bar-tab',
                                 selected_className='nav-bar-tab-selected'),
                         dcc.Tab(label='Predict & order', value='tab-2', className='nav-bar-tab',
                                 selected_className='nav-bar-tab-selected'),
                         dcc.Tab(label='Performance', value='tab-3', className='nav-bar-tab',
                                 selected_className='nav-bar-tab-selected'),
                         dcc.Tab(label='Model', value='tab-4', className='nav-bar-tab',
                                 selected_className='nav-bar-tab-selected'),
                         dcc.Tab(label='Backtest', value='tab-5', className='nav-bar-tab',
                                 selected_className='nav-bar-tab-selected'),
                         dcc.Tab(label='Trading212', value='tab-6', className='nav-bar-tab',
                                 selected_className='nav-bar-tab-selected'),
                     ]
                     ),
        ]
    ),

    html.Div(id="app-body")

    # html.Div(generate_table(df), style={'width': '50%'})

])


@app.callback(Output('app-body', 'children'), [Input('tabs', 'value')])
def render_content(tab):
    if tab == 'tab-1':
        return app_body_tab_overview
    elif tab == 'tab-2':
        return app_body_tab_predict
    elif tab == 'tab-3':
        pass
    elif tab == 'tab-4':
        return app_body_tab_model
    elif tab == 'tab-5':
        pass
    elif tab == 'tab-6':
        pass


if __name__ == '__main__':
    app.run_server(debug=True)
