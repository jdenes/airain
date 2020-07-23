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


app_body_tab_overview = [
    html.Div([
        "here, a plot of balance evolution (true v. theoretical)"
    ],
        className='app-body-balance-section'),
    html.Div([
        "here, a list of performances"
    ],
        className='app-body-perf-section'),
    html.Div([
        "here, a list of yesterday performances"
    ],
        className='app-body-yesterday-section'),
    html.Div([
        "here, a list of predictions for next day"
    ],
        className='app-body-pred-section'),
]


app_body_tab_predict = [
    html.Div(
        style={'width': '32%', 'height': '96.5%', 'display': 'inline-flex', 'flexFlow': 'column', 'padding': '10px 5px 10px 10px'},
        children=[
            html.Button("Update data", className='app-body-update-button'),
            html.Div("✔  Data are up to date", className='app-body-status-section'),
            html.Div("Showing the N last entry in PRICES to check date", className='app-body-prices-section'),
            html.Div("Showing the N last entry in NEWS to check date", className='app-body-news-section'),
        ]
    ),
    html.Div(
        style={'width': '32%', 'height': '96.5%', 'display': 'inline-flex', 'flexFlow': 'column', 'padding': '10px 5px 10px 5px'},
        children=[
            html.Button("Compute recommendations", className='app-body-update-button'),
            html.Div("⚠  Today's recommendation are not computed", className='app-body-status-section'),
            html.Div("Showing the N last recommendations if data is updated", className='app-body-news-section'),
        ]
    ),
    html.Div(
        style={'width': '32%', 'height': '96.5%', 'display': 'inline-flex', 'flexFlow': 'column', 'padding': '10px 10px 10px 5px'},
        children=[
            html.Button("Open trades", className='app-body-update-button'),
            html.Div("⚠  Orders have not been placed", className='app-body-status-section'),
            html.Div("Showing quantities if recommendation is ready", className='app-body-news-section'),
        ]
    ),
]


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
                         dcc.Tab(label='Recent performance', value='tab-3', className='nav-bar-tab',
                                 selected_className='nav-bar-tab-selected'),
                         dcc.Tab(label='Long term performance', value='tab-4', className='nav-bar-tab',
                                 selected_className='nav-bar-tab-selected'),
                         dcc.Tab(label='Trading212 status', value='tab-5', className='nav-bar-tab',
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
        return html.Div([
            html.H3('Tab content 3')
        ])
    elif tab == 'tab-4':
        return html.Div([
            html.H3('Tab content 4')
        ])


if __name__ == '__main__':
    app.run_server(debug=True)
