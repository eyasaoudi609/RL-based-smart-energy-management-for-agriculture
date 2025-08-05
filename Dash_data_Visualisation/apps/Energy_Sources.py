import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output,State
from dash.exceptions import PreventUpdate
import pandas as pd
import xgboost as xgb
import pickle
from app import app
import base64
import io
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from xgboost import XGBRegressor
import joblib
import datetime
import numpy as np


# Load the CSV data
data = pd.read_csv('C:\\Users\\saoudi\\Desktop\\My_Dash\\weekly_max_min.csv', delimiter=',')

# Convert the 'datetime' column to datetime type
data['datetime'] = pd.to_datetime(data['datetime'])
# Load the trained XGBoost model
modelPV = joblib.load('power_prediction_model.pkl')  # Replace with your model file path

# Group the data by week and calculate max and min for each feature
weekly_max_min = data.groupby(data['datetime'].dt.strftime('%U-%Y')).agg({
    'temperature_max': 'max',
    'temperature_min': 'min',
    'humidity_max': 'max',
    'humidity_min': 'min',
    'irradiation_max': 'max',
    'irradiation_min': 'min',
    'wind_speed_max': 'max',
    'wind_speed_min': 'min',
    'pressure_max': 'max',
    'pressure_min': 'min',
    'cell_temperature_max': 'max',
    'cell_temperature_min': 'min'
}).reset_index()
# Load the PV data from the CSV file
df_PV = pd.read_csv('C:\\Users\\saoudi\\Desktop\\My_Dash\\Dataset_PV_corrected.csv', delimiter=',' ,parse_dates=['datetime'])

# # Convert the 'datetime' column to datetime format if it's not already



# Background color for the plot
plot_bgcolor = '#d3e8c9'
# Load your trained XGBoost model
with open('xgboost_Biomass_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

model_filename = 'power_generation_model (1).pkl'
xgb_model = joblib.load(model_filename)
# Sample data for the donut chart (replace with your data)
biomass_data = pd.read_csv('C:\\Users\\saoudi\\Desktop\\My_Dash\\biomass_full_comp.csv', delimiter=';')
# Sample data (replace with your own data source)
# Create a list of unique biomass values for colors
# biomass_colors = biomass_data['Biomass type'].unique()
biomass_colors = {
    'Type1': 'blue',
    'Type2': 'red',
    'Type3': 'purple',
    'Type4': 'orange',
    'Type5': 'pink',
    # Add more biomass types and colors as needed
}
# Sample data (replace with your own data source)
data = {
    
'Biomass type': ['Olive wood', 'Pine chips', 'Pine sawdust', 'Eucalyptus bark', 'Land clearing wood', 'Wood residue', 'Reed canary grass', 'Sorghastrum grass', 'Sweet sorghum grass', 'Alfalfa straw', 'Barley straw', 'Corn straw', 'Oat straw', 'Rice straw', 'Wheat straw', 'Almond shells', 'Olive husks', 'Olive pits', 'Palm fibres-husks', 'Palm kernels', 'Rice husks', 'Soya husks', 'Sugar cane bagasse', 'Sunflower husks', 'Chicken litter', 'Meat-bone meal', 'Biomass mixture', 'Wood-agricultural residue', 'Wood-almond residue', 'Wood-almond residue', 'Wood-straw residue', 'Wood-straw residue', 'Currency shredded', 'Demolition wood', 'Furniture waste', 'Mixed waste paper', 'Greenhouse-plastic waste', 'Refuse-derived fuel', 'Sewage sludge', 'Wood yard waste'],
'M': [6.6, 7.6, 15.3, 12.0, 49.2, 26.4, 7.7, 11.3, 7.0, 9.3, 11.5, 7.4, 7.4, 7.6, 10.1, 7.2, 6.8, 6.1, 36.4, 11.0, 10.6, 6.3, 10.4, 9.1, 9.3, 2.5, 8.8, 30.3, 22.7, 22.7, 7.3, 7.3, 4.7, 16.3, 12.1, 8.8, 2.5, 4.2, 6.4, 38.1],
'VM': [79.6, 72.4, 83.1, 78.0, 69.7, 78.0, 73.4, 81.6, 77.2, 78.9, 76.2, 73.1, 80.5, 64.3, 74.8, 74.9, 79.0, 77.0, 72.8, 77.3, 62.8, 74.3, 85.5, 76.0, 47.8, 63.3, 69.4, 78.5, 77.2, 77.2, 75.51, 75.51, 82.9, 75.8, 83.0, 84.2, 62.6, 73.4, 48.0, 66.0],
'FC': [17.2, 21.6, 16.8, 17.2, 13.8, 16.6, 17.7, 14.2, 18.1, 15.8, 18.5, 19.2, 13.6, 15.6, 18.1, 21.8, 18.7, 19.9, 18.9, 17.5, 19.2, 20.3, 12.4, 20.9, 14.4, 12.7, 18.1, 18.2, 15.9, 15.9, 16.7, 16.7, 11.6, 17.3, 13.4, 7.5, 5.6, 0.5, 5.7, 13.6],
'A': [3.2, 6.0, 0.1, 4.8, 16.5, 5.4, 8.9, 4.2, 4.7, 5.3, 5.3, 7.7, 5.9, 20.1, 7.1, 3.3, 2.3, 3.1, 8.3, 5.2, 18.0, 5.4, 2.1, 3.1, 37.8, 24.0, 12.5, 3.3, 6.9, 6.9, 8.2, 8.2, 5.5, 6.9, 3.6, 8.3, 31.8, 26.1, 46.3, 20.4],
'C': [47.432, 49.632, 50.949, 46.3624, 42.3345, 48.6244, 45.0034, 47.3252, 47.3641, 47.2553, 46.7818, 44.9501, 45.9208, 40.0299, 45.8926, 48.6401, 48.85, 51.1632, 47.2255, 48.348, 40.426, 42.9484, 48.7542, 48.8376, 37.631, 43.548, 49.6125, 50.6708, 47.3879, 47.3879, 47.4606, 47.4606, 42.903, 48.1327, 49.9352, 47.9591, 48.3538, 39.7582, 27.3333, 41.5512],
'O': [43.4632, 38.07, 42.8571, 43.1256, 35.738, 39.6374, 38.8997, 42.152, 41.6461, 38.6376, 41.2892, 40.7043, 41.9686, 34.357, 40.5044, 41.0975, 41.1317, 38.1786, 36.7717, 37.446, 35.834, 44.3674, 42.9781, 41.667, 15.7366, 15.808, 28.9625, 39.8404, 39.5675, 39.5675, 38.097, 38.097, 43.5645, 37.8917, 40.2952, 36.8634, 11.1848, 27.1952, 17.9358, 32.1584],
'H': [5.2272, 5.734, 5.994, 5.4264, 5.01, 5.7706, 5.7393, 6.0354, 5.8133, 5.9661, 5.8714, 5.9072, 5.646, 4.5543, 5.6669, 5.9954, 6.0574, 6.3954, 6.0522, 6.162, 5.002, 6.3382, 5.874, 5.3295, 4.2296, 6.08, 5.775, 5.802, 5.4929, 5.4929, 5.7834, 5.7834, 5.9535, 5.9584, 5.8804, 6.6024, 7.6384, 5.7642, 3.9201, 4.776],
'N': [0.6776, 0.47, 0.0999, 0.2856, 0.334, 0.473, 1.3665, 0.2874, 0.3812, 2.6516, 0.6629, 0.6461, 0.4705, 0.799, 0.6503, 0.967, 1.5632, 1.0659, 1.3755, 2.5596, 0.656, 0.8514, 0.1958, 1.0659, 3.8564, 9.272, 2.3625, 0.3868, 0.5586, 0.5586, 0.3672, 0.3672, 1.7955, 1.0241, 0.2892, 0.1834, 1.023, 0.8129, 3.2757, 0.8756],
'S': [0.0, 0.094, 0.0, 0.0, 0.0835, 0.0946, 0.0911, 0.0, 0.0953, 0.1894, 0.0947, 0.0923, 0.0941, 0.1598, 0.1858, 0.0, 0.0977, 0.0969, 0.2751, 0.2844, 0.082, 0.0946, 0.0979, 0.0, 0.7464, 1.292, 0.7875, 0.0, 0.0931, 0.0931, 0.0918, 0.0918, 0.2835, 0.0931, 0.0, 0.0917, 0.0, 0.3695, 1.2351, 0.2388],


}


df = pd.DataFrame(data)



layout = html.Div([
    ######################################TITLE################################################
    html.Div([
        html.H1('Energy Sources Dashboard', style={'color': 'black', 'text-align': 'center', 'font-weight': 'bold'}),
    ], className='logo_title', style={'text-align': 'center'}),
    
#########################################SUBTITLE##############################################
    html.Div([
        html.H4('Biomass Composition Analysis', style={'font-size':'30px','text-align': 'center','color':'#62a484','font-weight': 'bold'}),
        ################################PIE CHART###################################################
    html.Div(
        dcc.Graph(id="biomass-pie-chart"),
            className='result-box1', style={'text-align': 'center','display': 'inline-block'}  # Apply className to the div wrapping the pie chart
        ),
    ##########################################DATA IN BRIEF###########################################
        html.Div([
        html.H1('Data in Brief', style={'color': '#D76400','display':'center','text-align': 'center',   'margin': '20px',  # Remove margin
            'padding': '0'}),
        
        # html.H2('The subject of this study revolves around the realm of energy, specifically focusing on renewable energy sources and the process of <span style="color: red;">biomass gasification</span> for power production. The data under scrutiny is sourced from a downdraft biomass gasification-power production plant, encompassing a diverse range of biomass materials and varying operational parameters. This data is organized and presented in the form of a table, serving as a crucial resource for analyzing the performance and efficiency of <span style="color: blue;">biomass gasification</span> as a sustainable energy generation method across different feedstock types and operational settings.', 
        #         style={'color': '#3D3635','display':'center','text-align': 'justify ','font-size':'20px'}),
                
    dcc.Markdown('''
The data is organized and presented in the form of a table, serving as a crucial resource for analyzing the performance and efficiency of <span style="color: blue;">biomass gasification</span> as a sustainable energy generation method across different feedstock types and operational settings.

Each case encompasses distinct elemental analysis compositions, including the proportions of **carbon (C)**, **oxygen (O)**, **hydrogen (H)**, **nitrogen (N)**, and **sulfur (S)**. Additionally, the cases present diverse proximate analysis compositions, which encompass **moisture content (M)**, **ash content (A)**, **volatile material content (VM)**, and **fixed carbon content (FC)**. Furthermore, the dataset incorporates critical operational parameters such as gasifier temperature and the air-to-fuel ratio. Lastly, it records the net output power generated in each case.

[To read more about the described data and download it, click here](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7575841/)
    ''', dangerously_allow_html=True, style={'color': 'black', 'text-align': 'justify','font-size':'20px'})
    ], className='result-box4', style={'text-align': 'right',
                                       'display': 'inline-block',
                                       'position': 'absolute',  # Change position to absolute
                                        'top': '120px',           # Adjust the top distance as needed
                                        'right': '50px', 'width': '900px' }  # Apply className to the div wrapping the pie chart
),
################################# BIOMASS TYPE SELECTION###########################################


        html.P("Select Biomass Type:", style={'color': '#D76400','text-align': 'center','font-weight': 'bold','font-size':'20px'}),
        dcc.Dropdown(
    id='biomass-type-dropdown',
    options=[{'label': t, 'value': t} for t in df['Biomass type']],
    value='Olive wood',  # Default selection
    clearable=False,
    style={
        'margin': '0 auto',  # Center the dropdown horizontally
        'margin-bottom': '200px',
        'width': '370px',  # Adjust the width as needed
        'font-size': '90%',
        'background-color': '#d3e8c9',
        'font-weight': 'bold',
        'font-size': '15px',
        'color':'white'
    },
    className='custom-dropdown'  # Apply a custom CSS class to the dropdown
)

    ]),
    
    
################################### COMPOSITION FOR PREDICTION  ####################################

    html.Div([
        html.Label("M:",style={'font-weight': 'bold','text-align': 'center'}),
        dcc.Input(id='input-m', type='number', value=23.0, style={'background-color': '#d3e8c9', 'color': 'black', 'border': '3px solid #62a484'}),
        html.Label("VM:",style={'font-weight': 'bold'}),
        dcc.Input(id='input-vm', type='number', value=80, style={'background-color': '#d3e8c9', 'color': 'black', 'border': '3px solid #62a484'} ),
        html.Label("FC:",style={'font-weight': 'bold'}),
        dcc.Input(id='input-fc', type='number', value=16, style={'background-color': '#d3e8c9', 'color': 'black', 'border': '3px solid #62a484'}),
        html.Label("A:",style={'font-weight': 'bold'}),
        dcc.Input(id='input-a', type='number', value=7, style={'background-color': '#d3e8c9', 'color': 'black', 'border': '3px solid #62a484'}),
        html.Label("C:",style={'font-weight': 'bold'}),
        dcc.Input(id='input-c', type='number', value=50, style={'background-color': '#d3e8c9', 'color': 'black', 'border': '3px solid #62a484'}),
        html.Label("O:",style={'font-weight': 'bold'}),
        dcc.Input(id='input-o', type='number', value=42, style={'background-color': '#d3e8c9', 'color': 'black', 'border': '3px solid #62a484'}),
        html.Label("H:",style={'font-weight': 'bold'}),
        dcc.Input(id='input-h', type='number', value=8, style={'background-color': '#d3e8c9', 'color': 'black', 'border': '3px solid #62a484'}),
        html.Label("N:",style={'font-weight': 'bold'}),
        dcc.Input(id='input-n', type='number', value=0.5, style={'background-color': '#d3e8c9', 'color': 'black', 'border': '3px solid #62a484'}),
        html.Label("S:",style={'font-weight': 'bold'}),
        dcc.Input(id='input-s', type='number', value=0.1, style={'background-color': '#d3e8c9', 'color': 'black', 'border': '3px solid #62a484'}),        
        html.Label("T (°C)",style={'font-weight': 'bold'}),
        dcc.Input(id='input-temperature', type='number', value=1200, style={'background-color': '#d3e8c9', 'color': 'black', 'border': '3px solid #62a484','marginBottom': '20px'}),
        html.Div([
        html.Div(id='prediction-output'),  # Large box for prediction and chart
     ]),
     
    ], className='result-box2',),  # Apply className to the div wrapping the elements

    # html.Div([html.Button('Submit', style={"height": "auto", "margin-bottom": "auto"},
    #                           id='button'), ]),

        

   ######################################## INTERACTIVE PLOT ###############################################

    html.Div([
    dcc.Graph(
        id='biomass-power-plot',
        style={'margin-right': '10px'},
        config={'displayModeBar': False},  # Hide the mode bar (optional)
    ),
    dcc.Dropdown(
        id='temperature-dropdown',
        options=[
            {'label': '600°C', 'value': 600},
            {'label': '900°C', 'value': 900},
            {'label': '1200°C', 'value': 1200},
            {'label': '1500°C', 'value': 1500}
        ],
        value=1200,  # Default temperature
        style={
            'text-align': 'left',
            'margin-top': '50px',
            'backgroundColor': '#E8CAD3',  # Change the background color
            'color': '#A2848D'  # Change the text color
        },
        # Define a custom CSS class for the dropdown options
        className='custom-dropdown-options'
    )
], className='result-box3', style={
    'text-align': 'right',
                                       'display': 'inline-block',
                                       'position': 'absolute',  # Change position to absolute
                                        'top': '930px',           # Adjust the top distance as needed
                                        'right': '50px', 'width': '900px'
}),


  html.Div([
    html.H4("PV Power Generation", style={'margin-bottom': '100px', 'margin-top': '100px', 'font-size': '30px', 'text-align': 'center', 'color': '#62a484', 'font-weight': 'bold'}),

    html.Div([
        html.Div([
            dcc.Dropdown(
                id='date-dropdown',
                options=[{'label': str(date), 'value': date} for date in df_PV['datetime'].dt.date.unique()],
                value=df_PV['datetime'].dt.date.max(),
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
                id='feature-dropdown',
                options=[{'label': col, 'value': col} for col in df_PV.columns[2:]],  # Exclude 'datetime' and 'power'
                value=df_PV.columns[2],
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
            dcc.Graph(id='feature-curve'),
        ], className='result-box3', style={
            'text-align': 'right',
            'display': 'inline-block',
            'position': 'absolute',  # Change position to absolute
            'right': '50px',
            'width': '900px',
            'z-index': '3',
           
        }),

        # Add the GIF in the corner of the box
        html.Div([
            html.Img(src='/assets/pv.gif', style={'position': 'absolute', 'margin-top': '60px', 'right': '150px', 'width': '150px', 'height': '150px', 'z-index': '4'})
        ]),
    ]),
]),
html.Div([
        html.H1('Data in Brief', style={'color': '#D76400','display':'center','text-align': 'center',   'margin': '20px',  # Remove margin
            'padding': '0'}),
        
        # html.H2('The subject of this study revolves around the realm of energy, specifically focusing on renewable energy sources and the process of <span style="color: red;">biomass gasification</span> for power production. The data under scrutiny is sourced from a downdraft biomass gasification-power production plant, encompassing a diverse range of biomass materials and varying operational parameters. This data is organized and presented in the form of a table, serving as a crucial resource for analyzing the performance and efficiency of <span style="color: blue;">biomass gasification</span> as a sustainable energy generation method across different feedstock types and operational settings.', 
        #         style={'color': '#3D3635','display':'center','text-align': 'justify ','font-size':'20px'}),
                
    dcc.Markdown('''
 We have exploited the potential of PV systems to drive power
generation. The duration of nine months for data collection was carefully chosen to en-
capsulate different seasons, weather patterns, and solar angles. 

                 
This extended timeframe provided us with a global view of how PV systems respond to varying conditions and
fluctuations in solar irradiance over an annual cycle.
                 
            

                 
The data gathered spans from **February 25, 2022**, to **September 27, 2022**, with measurements taken at 5-minute intervals. 
                 
These measurements include **power**, **temperature**, **humidity**, **irradiation**, **wind speed**, **pressure**, and **cell temperature**.
    ''', dangerously_allow_html=True, style={'color': 'black', 'text-align': 'justify','font-size':'20px'})
    ],  style={'margin-top':'120px', 'margin-left':'300px','text-align': 'center',
  'border-radius': '10px','height':'700px',
  'border': '2px solid black','width':'500px', 'padding': '20px', 'margin-bottom':'50px'}  # Apply className to the div wrapping the pie chart
),
html.Div([
        html.H1('Weekly PV Data', style={'color': '#D76400','display':'center','text-align': 'center',   'margin-top': '50px', 'margin-bottom': '0' , # Remove margin
            'padding': '0'}),

        
html.Div([
  
    dcc.Graph(id='weekly-power-plot'),

    dcc.Interval(
        id='interval-component',
        interval=1000*60*60*24*7,  # Update weekly (7 days)
        n_intervals=0
    ),
],className='result-box5',style={
            'text-align': 'right',
            'display': 'inline-block',
            'position': 'absolute',  # Change position to absolute
            'left': '300px',
            'margin-top':'100px'
           
        
        }),
 html.Div([
    html.Div(
        dcc.Graph(
            id='weekly-max-min-chart',
            figure=px.bar(
                weekly_max_min,
                x='datetime',
                y=['temperature_max', 'temperature_min', 'humidity_max', 'humidity_min',
                   'irradiation_max', 'irradiation_min', 'wind_speed_max', 'wind_speed_min',
                   'pressure_max', 'pressure_min', 'cell_temperature_max', 'cell_temperature_min'],
                barmode='group',
                title='Weekly Max and Min Values for Each Feature',
            ).update_layout(
                plot_bgcolor=plot_bgcolor,  # Set background color for the plot
                paper_bgcolor=plot_bgcolor,  # Set background color for the entire plot
                font=dict(color='black')
            ),
            config={'displayModeBar': False},  # Hide the display mode bar
        ),
        className='result-box6',style={
            'text-align': 'right',
            'display': 'inline-block',
            'position': 'absolute',  # Change position to absolute
            'left': '1100px',
            'margin-top':'100px'
            
           
        
        },
    )
])

]),html.Div([
    dcc.Graph(id='monthly-power-plot'),
],
        className='result-box7',style={
            'text-align': 'right',
            'display': 'inline-block',
            'position': 'absolute',  # Change position to absolute
            'left': '300px',
            'margin-top':'700px'
            
           
        
        },)
        ,
        html.Div([
    dcc.Upload(
        id='upload-data',
        children=html.Button('Upload Historical Data CSV File'),
        multiple=False
    ),
    dcc.Graph(id='power-prediction-plotPV'),
],className='result-box6',style={
            'text-align': 'right',
            'display': 'inline-block',
            'position': 'absolute',  # Change position to absolute
            'left': '1100px',
            'top':'3400px','height':'550px'
            
           
        
        })


])


########################################### CALLBACKS ####################################################
@app.callback(
    Output('prediction-output', 'children'),
    Input('input-m', 'value'),
    Input('input-vm', 'value'),
    Input('input-fc', 'value'),
    Input('input-a', 'value'),
    Input('input-c', 'value'),
    Input('input-o', 'value'),
    Input('input-h', 'value'),
    Input('input-n', 'value'),
    Input('input-s', 'value'),
    Input('input-temperature', 'value')
)
def update_prediction(m, vm, fc, a, c, o, h, n, s, temperature):
    input_data = pd.DataFrame({
        'M': [m],
        'VM': [vm],
        'FC': [fc],
        'A': [a],
        'C': [c],
        'O': [o],
        'H': [h],
        'N': [n],
        'S': [s],
        'T (?C)': [temperature]
    })
    prediction = model.predict(input_data)
    return html.Div(f'Predicted biomass power : {prediction[0]}', style={'color': 'brown','font-weight': 'bold'})


@app.callback(
    Output("biomass-pie-chart", "figure"),
    Input("biomass-type-dropdown", "value")
)
def generate_pie_chart(selected_biomass_type):
    # Filter data for the selected biomass type
    selected_data = df[df['Biomass type'] == selected_biomass_type]

    # Calculate composition percentages
    composition_columns = ['M', 'VM', 'FC', 'A', 'C', 'O', 'H', 'N', 'S']
    composition_percentages = [selected_data[column].values[0] for column in composition_columns]

    # Create a pie chart
    fig = px.pie(names=composition_columns, values=composition_percentages, hole=0.3,
                 title=f'Composition of {selected_biomass_type}')

    # Modify the background color of the chart
    fig.update_layout(
        paper_bgcolor='#d3e8c9',  # Background color of the plot area
        plot_bgcolor='#d3e8c9'    # Background color of the entire chart
    )

    return fig

@app.callback(
    Output('biomass-power-plot', 'figure'),
    [Input('temperature-dropdown', 'value')]
)
def update_plot(selected_temperature):
    filtered_df = biomass_data[biomass_data['T (°C)'] == selected_temperature]

    fig = px.bar(
        filtered_df,
        x='Biomass type',
        y='Wnet (kW)',
        title=f'Produced Power at {selected_temperature}°C for each biomass type',
        color='Biomass type',  # Use the 'Biomass' column for coloring bars
        color_discrete_map=biomass_colors, 
        
    )
    fig.update_layout(
        paper_bgcolor='#d3e8c9',  # Background color of the plot area
        plot_bgcolor='#d3e8c9'    # Background color of the entire chart
    )
    return fig

# Callback to update the histograms
@app.callback(
    Output('feature-curve', 'figure'),
    Input('date-dropdown', 'value'),
    Input('feature-dropdown', 'value')
)
def update_curve(selected_date, selected_feature):
    selected_date = pd.to_datetime(selected_date)  # Convert to Pandas Timestamp
    filtered_df = df_PV[df_PV['datetime'].dt.date == selected_date.date()]

    fig = px.line(filtered_df, x='datetime', y=selected_feature)
    fig.update_xaxes(title_text='Datetime', title_font=dict(color='#A2848D', size=20))
    fig.update_yaxes(title_text=selected_feature, title_font=dict(color='#A2848D', size=20))
    fig.update_layout(
        paper_bgcolor='#d3e8c9',  # Background color of the plot area
        plot_bgcolor='#d3e8c9',    # Background color of the entire chart
        xaxis_title=dict(text='Datetime', font=dict(color='#A2848D', size=20), standoff=30),
        yaxis_title=dict(text=selected_feature, font=dict(color='#A2848D', size=20), standoff=30),
        title=f'{selected_feature} on {selected_date.date()}',  # Main title with custom size
        title_font=dict(size=24),  # Adjust the font size of the main title
        title_x=0.5,               # Center the title horizontally
        title_y=0.95               # Adjust the vertical position of the title
    )
    return fig




# Define the callback function to update the weekly power plot
@app.callback(
    Output('weekly-power-plot', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_weekly_power_plot(n_intervals):
    # Resample the data to a weekly frequency and calculate the sum
    weekly_power = df_PV.resample('W', on='datetime').sum()
    
    # Create a Plotly figure
    fig = px.bar(weekly_power, x=weekly_power.index, y='power', labels={'power': 'Power Produced (kW)'})
    fig.update_layout(
        title='Weekly Power Production',
        xaxis_title='Week',
        yaxis_title='Power Produced (kW)',
        xaxis={'type': 'category'},
    )
    fig.update_layout(
        paper_bgcolor='#d3e8c9',  # Background color of the plot area
        plot_bgcolor='#d3e8c9'    # Background color of the entire chart
    )
    

    return fig

@app.callback(
    Output('monthly-power-plot', 'figure'),
    Input('monthly-power-plot', 'relayoutData')
)
def update_power_plot(relayoutData):
    df_PV['datetime'] = pd.to_datetime(df_PV['datetime'])
    
    # Create a 'Month' column for grouping by month
    df_PV['Month'] = df_PV['datetime'].dt.to_period('M')

    # Add a new column for the string representation of 'Month'
    df_PV['Month_str'] = df_PV['Month'].dt.strftime('%Y-%m')

    # Group the data by the new 'Month_str' column and calculate the total power for each month
    monthly_power_production = df_PV.groupby('Month_str')['power'].sum().reset_index()

    # Create a bar chart of the monthly power production using the 'Month_str' column
    fig = px.bar(monthly_power_production, x='Month_str', y='power', labels={'power': 'Monthly Power Production'},
                 color='Month_str')  # Use 'Month_str' for coloring
    
    fig.update_layout(title='Monthly Power Production')
    fig.update_layout(
        paper_bgcolor='#d3e8c9',  # Background color of the plot area
        plot_bgcolor='#d3e8c9'  # Background color of the entire chart
    )

    return fig


@app.callback(
    Output('power-prediction-plotPV', 'figure'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    State('upload-data', 'last_modified')
)
def power_prediction_plot(contents, filename, last_modified):
    if contents is None:
        raise dash.exceptions.PreventUpdate

    # Decode and read the uploaded CSV file
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    historical_data = pd.read_csv(io.StringIO(decoded.decode('latin1')))

    # Perform power prediction based on the historical data (similar to your existing code)
    forecast_period_hours = 24
    end_datetime = datetime.datetime.now()
    forecast_datetime = end_datetime + pd.to_timedelta(np.arange(1, forecast_period_hours + 1), unit='H')

    forecast_data = pd.DataFrame({'datetime': forecast_datetime, 'power': None})

    for i, forecast_time in enumerate(forecast_datetime):
        forecast_input_data = historical_data.copy()
        forecast_input_data['hour'] = forecast_time.hour
        forecast_input_data['minute'] = forecast_time.minute
        forecast_power = modelPV.predict(forecast_input_data)
        forecast_data.at[i, 'power'] = forecast_power[0]

    # Create a Plotly figure for the predicted power values
    fig = px.line(forecast_data, x='datetime', y='power', title='Predicted Power for the Next 24 Hours')
    fig.update_xaxes(title='Datetime')
    fig.update_yaxes(title='Power')
    fig.update_layout(xaxis=dict(
        tickmode='array',
        tickvals=list(range(0, 24, 2)),
        ticktext=[str(i) for i in range(0, 24, 2)]
    ))
    fig.update_layout(
        paper_bgcolor='#d3e8c9',  # Background color of the plot area
        plot_bgcolor='#d3e8c9'  # Background color of the entire chart
    )
    return fig
