from dash import html, dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
from app import app
import pandas as pd
import dash_table
import dash
import plotly.express as px

# Load the datasets
df_cleaning = pd.read_csv('C:\\Users\\saoudi\\Desktop\\My_Dash\\cleaning_system_10min.csv', delimiter=',',parse_dates=['timestamp'])
df_cooling = pd.read_csv('C:\\Users\\saoudi\\Desktop\\My_Dash\\cooling_system_10min.csv', delimiter=',',parse_dates=['timestamp'])
df_lightning = pd.read_csv('C:\\Users\\saoudi\\Desktop\\My_Dash\\lightning_system_10min.csv', delimiter=',',parse_dates=['timestamp'])
df_vacuum_pump = pd.read_csv('C:\\Users\\saoudi\\Desktop\\My_Dash\\vacuum_pump_10min.csv', delimiter=',',parse_dates=['timestamp'])
df_water_pump = pd.read_csv('C:\\Users\\saoudi\\Desktop\\My_Dash\\Waterpump.csv', delimiter=',',parse_dates=['Date'])
df_coldroom = pd.read_csv('C:\\Users\\saoudi\\Desktop\\My_Dash\\coldroom.csv', delimiter=',',parse_dates=['Date'])

def generate_data_table(id, columns, data):
    return dash_table.DataTable(
        id=id,
        columns=[{"name": col, "id": col} for col in columns],
        data=data.to_dict('records'),
        style_table={'height': '650px', 'overflowY': 'scroll', 'backgroundColor': '#c9e8d3'},
        style_cell={'backgroundColor': '#c9e8d3'},
        style_data_conditional=[
            {'if': {'row_index': 'odd'}, 'backgroundColor': '#62a484'},
            {'if': {'row_index': 'even'}, 'backgroundColor': '#d3e8c9'},
            {'if': {'column_editable': True}, 'border': '1px solid #62a484'},
            {'if': {'row_index': 'odd'}, 'border-top': '2px solid #62a484'},
            {'if': {'row_index': 'even'}, 'border-top': '2px solid #62a484'},
        ]
    )

layout = html.Div([
    
    ######################################TITLE################################################
    html.Div([
        html.H1('Appliances', style={'color': 'black', 'text-align': 'center', 'font-weight': 'bold'}),
    ], className='logo_title', style={'text-align': 'center','margin-bottom':'100px'}),
    
    html.Div(
        [
            html.H4('Cleaning System',
                    style={'font-size': '20px', 'text-align': 'center', 'color': '#000000', 'font-weight': 'bold'}),
            generate_data_table('table1', df_cleaning.columns, df_cleaning)
        ],
        className='result-box8',style={'margin-bottom':'50px'}),

    html.Div(
        [
            html.H4('Cooling System',
                    style={'font-size': '20px', 'text-align': 'center', 'color': '#000000', 'font-weight': 'bold'}),
            generate_data_table('table2', df_cooling.columns, df_cooling)
        ],
        className='result-box8',style={'margin-bottom':'50px'}),

    html.Div(
        [
            html.H4('Lightning System',
                    style={'font-size': '20px', 'text-align': 'center', 'color': '#000000', 'font-weight': 'bold'}),
            generate_data_table('table3', df_lightning.columns, df_lightning)
        ],
        className='result-box8',style={'margin-bottom':'50px'}),

    html.Div(
        [
            html.H4('Vacuum Pump',
                    style={'font-size': '20px', 'text-align': 'center', 'color': '#000000', 'font-weight': 'bold'}),
            generate_data_table('table4', df_vacuum_pump.columns, df_vacuum_pump)
        ],
        className='result-box8',style={'margin-bottom':'50px'}),

    html.Div(
        [
            html.H4('Water Pump',
                    style={'font-size': '20px', 'text-align': 'center', 'color': '#000000', 'font-weight': 'bold'}),
            generate_data_table('table5', df_water_pump.columns, df_water_pump)
        ],
        className='result-box8',style={'margin-bottom':'50px'}),

    html.Div(
        [
            html.H4('Coldroom',
                    style={'font-size': '20px', 'text-align': 'center', 'color': '#000000', 'font-weight': 'bold'}),
            generate_data_table('table6', df_coldroom.columns, df_coldroom)
        ],
        className='result-box8',style={'margin-bottom':'50px'}),

html.Div([
        html.Div([
            dcc.Dropdown(
                id='date-dropdown1',
                options=[{'label': str(date), 'value': date} for date in df_cleaning['timestamp'].dt.date.unique()],
                value=df_cleaning['timestamp'].dt.date.max(),
                style={
                    'text-align': 'left',
                    'margin-top': '50px',
                    'width': '370px',
                    'z-index': '2',
                    'font-size': '90%',
                    'background-color': '#d3e8c9',
                    'font-weight': 'bold',
                    'font-size': '15px',
                    'backgroundColor': '#d3e8c9',  # Change the background color
                    'color': '#A2848D',
                    'box-shadow': '0 0 0 3px #A2848D'  # Change the text color
                },
                className='custom-dropdown-options',
            ),
            # Dropdown to select a feature
            dcc.Dropdown(
                id='feature-dropdown1',
                options=[{'label': col, 'value': col} for col in df_cleaning.columns[2:]],  # Exclude 'datetime' and 'power'
                value=df_cleaning.columns[2],
                style={
                    'text-align': 'left',
                    'font-size': '90%',
                    'font-weight': 'bold',
                    'font-size': '15px',
                    'color': 'white',  # Change the background color
                    'backgroundColor': '#d3e8c9',  # Change the background color
                    'margin-bottom': '50px',
                    'margin-top': '20px',
                    'width': '370px',
                    'box-shadow': '0 0 0 3px #A2848D',
                    'color': '#A2848D',
                },
            ),
            # Line chart
            dcc.Graph(id='feature-curve1'),
        ], className='result-box3', style={
            'text-align': 'right',
            'display': 'inline-block',
            'position': 'absolute',  # Change position to absolute
            'right': '50px',
            'width': '900px',
            'height':'800px',
            'top':'130px',
            'z-index': '3',
           
        }),

html.Div([
        html.Div([
            dcc.Dropdown(
                id='date-dropdown2',
                options=[{'label': str(date), 'value': date} for date in df_cooling['timestamp'].dt.date.unique()],
                value=df_cooling['timestamp'].dt.date.max(),
                style={
                    'text-align': 'left',
                    'margin-top': '50px',
                    'width': '370px',
                    'z-index': '2',
                    'font-size': '90%',
                    'background-color': '#d3e8c9',
                    'font-weight': 'bold',
                    'font-size': '15px',
                    'backgroundColor': '#d3e8c9',  # Change the background color
                    'color': '#A2848D',
                    'box-shadow': '0 0 0 3px #A2848D'  # Change the text color
                },
                className='custom-dropdown-options',
            ),
            # Dropdown to select a feature
            dcc.Dropdown(
                id='feature-dropdown2',
                options=[{'label': col, 'value': col} for col in df_cooling.columns[2:]],  # Exclude 'datetime' and 'power'
                value=df_cooling.columns[2],
                style={
                    'text-align': 'left',
                    'font-size': '90%',
                    'font-weight': 'bold',
                    'font-size': '15px',
                    'color': 'white',  # Change the background color
                    'backgroundColor': '#d3e8c9',  # Change the background color
                    'margin-bottom': '50px',
                    'margin-top': '20px',
                    'width': '370px',
                    'box-shadow': '0 0 0 3px #A2848D',
                    'color': '#A2848D',
                },
            ),
            # Line chart
            dcc.Graph(id='feature-curve2'),
        ], className='result-box3', style={
            'text-align': 'right',
            'display': 'inline-block',
            'position': 'absolute',  # Change position to absolute
            'right': '50px',
            'width': '900px',
            'height':'800px',
            'top':'985px',
            'z-index': '3',
           
        }),
])



]),
html.Div([
        html.Div([
            dcc.Dropdown(
                id='date-dropdown3',
                options=[{'label': str(date), 'value': date} for date in df_vacuum_pump['timestamp'].dt.date.unique()],
                value=df_vacuum_pump['timestamp'].dt.date.max(),
                style={
                    'text-align': 'left',
                    'margin-top': '50px',
                    'width': '370px',
                    'z-index': '2',
                    'font-size': '90%',
                    'background-color': '#d3e8c9',
                    'font-weight': 'bold',
                    'font-size': '15px',
                    'backgroundColor': '#d3e8c9',  # Change the background color
                    'color': '#A2848D',
                    'box-shadow': '0 0 0 3px #A2848D'  # Change the text color
                },
                className='custom-dropdown-options',
            ),
            # Dropdown to select a feature
            dcc.Dropdown(
                id='feature-dropdown3',
                options=[{'label': col, 'value': col} for col in df_vacuum_pump.columns[2:]],  # Exclude 'datetime' and 'power'
                value=df_vacuum_pump.columns[2],
                style={
                    'text-align': 'left',
                    'font-size': '90%',
                    'font-weight': 'bold',
                    'font-size': '15px',
                    'color': 'white',  # Change the background color
                    'backgroundColor': '#d3e8c9',  # Change the background color
                    'margin-bottom': '50px',
                    'margin-top': '20px',
                    'width': '370px',
                    'box-shadow': '0 0 0 3px #A2848D',
                    'color': '#A2848D',
                },
            ),
            # Line chart
            dcc.Graph(id='feature-curve3'),
        ], className='result-box3', style={
            'text-align': 'right',
            'display': 'inline-block',
            'position': 'absolute',  # Change position to absolute
            'right': '50px',
            'width': '900px',
            'height':'800px',
            'top':'1835px',
            'z-index': '3',
           
        }),



]),
html.Div([
        html.Div([
            dcc.Dropdown(
                id='date-dropdown4',
                options=[{'label': str(date), 'value': date} for date in df_vacuum_pump['timestamp'].dt.date.unique()],
                value=df_vacuum_pump['timestamp'].dt.date.max(),
                style={
                    'text-align': 'left',
                    'margin-top': '50px',
                    'width': '370px',
                    'z-index': '2',
                    'font-size': '90%',
                    'background-color': '#d3e8c9',
                    'font-weight': 'bold',
                    'font-size': '15px',
                    'backgroundColor': '#d3e8c9',  # Change the background color
                    'color': '#A2848D',
                    'box-shadow': '0 0 0 3px #A2848D'  # Change the text color
                },
                className='custom-dropdown-options',
            ),
            # Dropdown to select a feature
            dcc.Dropdown(
                id='feature-dropdown4',
                options=[{'label': col, 'value': col} for col in df_vacuum_pump.columns[2:]],  # Exclude 'datetime' and 'power'
                value=df_vacuum_pump.columns[2],
                style={
                    'text-align': 'left',
                    'font-size': '90%',
                    'font-weight': 'bold',
                    'font-size': '15px',
                    'color': 'white',  # Change the background color
                    'backgroundColor': '#d3e8c9',  # Change the background color
                    'margin-bottom': '50px',
                    'margin-top': '20px',
                    'width': '370px',
                    'box-shadow': '0 0 0 3px #A2848D',
                    'color': '#A2848D',
                },
            ),
            # Line chart
            dcc.Graph(id='feature-curve4'),
        ], className='result-box3', style={
            'text-align': 'right',
            'display': 'inline-block',
            'position': 'absolute',  # Change position to absolute
            'right': '50px',
            'width': '900px',
            'height':'800px',
            'top':'2685px',
            'z-index': '3',
           
        }),



]),
html.Div([
        html.Div([
            dcc.Dropdown(
                id='date-dropdown5',
                options=[{'label': str(date), 'value': date} for date in df_water_pump['Date'].dt.date.unique()],
                value=df_water_pump['Date'].dt.date.max(),
                style={
                    'text-align': 'left',
                    'margin-top': '50px',
                    'width': '370px',
                    'z-index': '2',
                    'font-size': '90%',
                    'background-color': '#d3e8c9',
                    'font-weight': 'bold',
                    'font-size': '15px',
                    'backgroundColor': '#d3e8c9',  # Change the background color
                    'color': '#A2848D',
                    'box-shadow': '0 0 0 3px #A2848D'  # Change the text color
                },
                className='custom-dropdown-options',
            ),
            # Dropdown to select a feature
            dcc.Dropdown(
                id='feature-dropdown5',
                options=[{'label': col, 'value': col} for col in df_water_pump.columns[2:]],  # Exclude 'datetime' and 'power'
                value=df_water_pump.columns[2],
                style={
                    'text-align': 'left',
                    'font-size': '90%',
                    'font-weight': 'bold',
                    'font-size': '15px',
                    'color': 'white',  # Change the background color
                    'backgroundColor': '#d3e8c9',  # Change the background color
                    'margin-bottom': '50px',
                    'margin-top': '20px',
                    'width': '370px',
                    'box-shadow': '0 0 0 3px #A2848D',
                    'color': '#A2848D',
                },
            ),
            # Line chart
            dcc.Graph(id='feature-curve5'),
        ], className='result-box3', style={
            'text-align': 'right',
            'display': 'inline-block',
            'position': 'absolute',  # Change position to absolute
            'right': '50px',
            'width': '900px',
            'height':'800px',
            'top':'3535px',
            'z-index': '3',
           
        }),



]),
html.Div([
        html.Div([
            dcc.Dropdown(
                id='date-dropdown6',
                options=[{'label': str(date), 'value': date} for date in df_coldroom['Date'].dt.date.unique()],
                value=df_coldroom['Date'].dt.date.max(),
                style={
                    'text-align': 'left',
                    'margin-top': '50px',
                    'width': '370px',
                    'z-index': '2',
                    'font-size': '90%',
                    'background-color': '#d3e8c9',
                    'font-weight': 'bold',
                    'font-size': '15px',
                    'backgroundColor': '#d3e8c9',  # Change the background color
                    'color': '#A2848D',
                    'box-shadow': '0 0 0 3px #A2848D'  # Change the text color
                },
                className='custom-dropdown-options',
            ),
            # Dropdown to select a feature
            dcc.Dropdown(
                id='feature-dropdown6',
                options=[{'label': col, 'value': col} for col in df_coldroom.columns[2:]],  # Exclude 'datetime' and 'power'
                value=df_coldroom.columns[2],
                style={
                    'text-align': 'left',
                    'font-size': '90%',
                    'font-weight': 'bold',
                    'font-size': '15px',
                    'color': 'white',  # Change the background color
                    'backgroundColor': '#d3e8c9',  # Change the background color
                    'margin-bottom': '50px',
                    'margin-top': '20px',
                    'width': '370px',
                    'box-shadow': '0 0 0 3px #A2848D',
                    'color': '#A2848D',
                },
            ),
            # Line chart
            dcc.Graph(id='feature-curve6'),
        ], className='result-box3', style={
            'text-align': 'right',
            'display': 'inline-block',
            'position': 'absolute',  # Change position to absolute
            'right': '50px',
            'width': '900px',
            'height':'800px',
            'top':'4385px',
            'z-index': '3',
           
        }),



]),
])










# Define a callback to update the plot based on user input or changes
# Callback to update the histograms cleaning system
@app.callback(
    Output('feature-curve1', 'figure'),
    Input('date-dropdown1', 'value'),
    Input('feature-dropdown1', 'value')
)
def update_curve(selected_date, selected_feature):
    selected_date = pd.to_datetime(selected_date)  # Convert to Pandas Timestamp
    filtered_df = df_cleaning[df_cleaning['timestamp'].dt.date == selected_date.date()]

    fig = px.line(filtered_df, x='timestamp', y=selected_feature)
    fig.update_xaxes(title_text='timestamp', title_font=dict(color='#A2848D', size=20))
    fig.update_yaxes(title_text=selected_feature, title_font=dict(color='#A2848D', size=20))
    fig.update_layout(
        paper_bgcolor='#d3e8c9',  # Background color of the plot area
        plot_bgcolor='#d3e8c9',    # Background color of the entire chart
        xaxis_title=dict(text='timestamp', font=dict(color='#A2848D', size=20), standoff=30),
        yaxis_title=dict(text=selected_feature, font=dict(color='#A2848D', size=20), standoff=30),
        title=f'{selected_feature} on {selected_date.date()}',  # Main title with custom size
        title_font=dict(size=24),  # Adjust the font size of the main title
        title_x=0.5,               # Center the title horizontally
        title_y=0.95               # Adjust the vertical position of the title
    )
    return fig

# Callback to update the histograms cleaning system
@app.callback(
    Output('feature-curve2', 'figure'),
    Input('date-dropdown2', 'value'),
    Input('feature-dropdown2', 'value')
)
def update_curve(selected_date, selected_feature):
    selected_date = pd.to_datetime(selected_date)  # Convert to Pandas Timestamp
    filtered_df = df_cooling[df_cooling['timestamp'].dt.date == selected_date.date()]

    fig = px.line(filtered_df, x='timestamp', y=selected_feature)
    fig.update_xaxes(title_text='timestamp', title_font=dict(color='#A2848D', size=20))
    fig.update_yaxes(title_text=selected_feature, title_font=dict(color='#A2848D', size=20))
    fig.update_layout(
        paper_bgcolor='#d3e8c9',  # Background color of the plot area
        plot_bgcolor='#d3e8c9',    # Background color of the entire chart
        xaxis_title=dict(text='timestamp', font=dict(color='#A2848D', size=20), standoff=30),
        yaxis_title=dict(text=selected_feature, font=dict(color='#A2848D', size=20), standoff=30),
        title=f'{selected_feature} on {selected_date.date()}',  # Main title with custom size
        title_font=dict(size=24),  # Adjust the font size of the main title
        title_x=0.5,               # Center the title horizontally
        title_y=0.95               # Adjust the vertical position of the title
    )
    return fig


@app.callback(
    Output('feature-curve3', 'figure'),
    Input('date-dropdown3', 'value'),
    Input('feature-dropdown3', 'value')
)
def update_curve(selected_date, selected_feature):
    selected_date = pd.to_datetime(selected_date)  # Convert to Pandas Timestamp
    filtered_df = df_lightning[df_lightning['timestamp'].dt.date == selected_date.date()]

    fig = px.line(filtered_df, x='timestamp', y=selected_feature)
    fig.update_xaxes(title_text='timestamp', title_font=dict(color='#A2848D', size=20))
    fig.update_yaxes(title_text=selected_feature, title_font=dict(color='#A2848D', size=20))
    fig.update_layout(
        paper_bgcolor='#d3e8c9',  # Background color of the plot area
        plot_bgcolor='#d3e8c9',    # Background color of the entire chart
        xaxis_title=dict(text='timestamp', font=dict(color='#A2848D', size=20), standoff=30),
        yaxis_title=dict(text=selected_feature, font=dict(color='#A2848D', size=20), standoff=30),
        title=f'{selected_feature} on {selected_date.date()}',  # Main title with custom size
        title_font=dict(size=24),  # Adjust the font size of the main title
        title_x=0.5,               # Center the title horizontally
        title_y=0.95               # Adjust the vertical position of the title
    )
    return fig


@app.callback(
    Output('feature-curve4', 'figure'),
    Input('date-dropdown4', 'value'),
    Input('feature-dropdown4', 'value')
)
def update_curve(selected_date, selected_feature):
    selected_date = pd.to_datetime(selected_date)  # Convert to Pandas Timestamp
    filtered_df = df_vacuum_pump[df_vacuum_pump['timestamp'].dt.date == selected_date.date()]

    fig = px.line(filtered_df, x='timestamp', y=selected_feature)
    fig.update_xaxes(title_text='timestamp', title_font=dict(color='#A2848D', size=20))
    fig.update_yaxes(title_text=selected_feature, title_font=dict(color='#A2848D', size=20))
    fig.update_layout(
        paper_bgcolor='#d3e8c9',  # Background color of the plot area
        plot_bgcolor='#d3e8c9',    # Background color of the entire chart
        xaxis_title=dict(text='timestamp', font=dict(color='#A2848D', size=20), standoff=30),
        yaxis_title=dict(text=selected_feature, font=dict(color='#A2848D', size=20), standoff=30),
        title=f'{selected_feature} on {selected_date.date()}',  # Main title with custom size
        title_font=dict(size=24),  # Adjust the font size of the main title
        title_x=0.5,               # Center the title horizontally
        title_y=0.95               # Adjust the vertical position of the title
    )
    return fig




@app.callback(
    Output('feature-curve5', 'figure'),
    Input('date-dropdown5', 'value'),
    Input('feature-dropdown5', 'value')
)
def update_curve(selected_date, selected_feature):
    selected_date = pd.to_datetime(selected_date)  # Convert to Pandas Timestamp
    filtered_df = df_water_pump[df_water_pump['Date'].dt.date == selected_date.date()]

    fig = px.line(filtered_df, x='Date', y=selected_feature)
    fig.update_xaxes(title_text='Date', title_font=dict(color='#A2848D', size=20))
    fig.update_yaxes(title_text=selected_feature, title_font=dict(color='#A2848D', size=20))
    fig.update_layout(
        paper_bgcolor='#d3e8c9',  # Background color of the plot area
        plot_bgcolor='#d3e8c9',    # Background color of the entire chart
        xaxis_title=dict(text='Date', font=dict(color='#A2848D', size=20), standoff=30),
        yaxis_title=dict(text=selected_feature, font=dict(color='#A2848D', size=20), standoff=30),
        title=f'{selected_feature} on {selected_date.date()}',  # Main title with custom size
        title_font=dict(size=24),  # Adjust the font size of the main title
        title_x=0.5,               # Center the title horizontally
        title_y=0.95               # Adjust the vertical position of the title
    )
    return fig



@app.callback(
    Output('feature-curve6', 'figure'),
    Input('date-dropdown6', 'value'),
    Input('feature-dropdown6', 'value')
)
def update_curve(selected_date, selected_feature):
    selected_date = pd.to_datetime(selected_date)  # Convert to Pandas Timestamp
    filtered_df = df_coldroom[df_coldroom['Date'].dt.date == selected_date.date()]

    fig = px.line(filtered_df, x='Date', y=selected_feature)
    fig.update_xaxes(title_text='Date', title_font=dict(color='#A2848D', size=20))
    fig.update_yaxes(title_text=selected_feature, title_font=dict(color='#A2848D', size=20))
    fig.update_layout(
        paper_bgcolor='#d3e8c9',  # Background color of the plot area
        plot_bgcolor='#d3e8c9',    # Background color of the entire chart
        xaxis_title=dict(text='Date', font=dict(color='#A2848D', size=20), standoff=30),
        yaxis_title=dict(text=selected_feature, font=dict(color='#A2848D', size=20), standoff=30),
        title=f'{selected_feature} on {selected_date.date()}',  # Main title with custom size
        title_font=dict(size=24),  # Adjust the font size of the main title
        title_x=0.5,               # Center the title horizontally
        title_y=0.95               # Adjust the vertical position of the title
    )
    return fig