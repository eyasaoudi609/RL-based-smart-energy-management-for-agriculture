from dash import html
from dash import dcc


layout = html.Div([
    html.Div([
        html.Div([
            html.P(
                 "Smart Green", style={"color": "#62a484",
                                "font-size": "100px",
                                'margin-left': 'auto',
                                'margin-right': 'auto',
                                'margin-top': 'auto',
                                'margin-bottom': 'auto',
                                'text-align': 'center',
                                'position': 'absolute',
                                'top': '0',
                                'bottom': '0',
                                'left': '0',
                                'right': '0'}
        ),
            html.P([html.P(dcc.Markdown('''Welcome to the **SmartGreen Farmscape **!''',
                                        style={"color": "#073B3A",
                                               "font-size": "50px",
                                               'margin-left': '200px',
                                               'margin-top': '150px',
                                               'margin-bottom': '0',
                                               'text-align': 'center'
                                               })),
                    
                    ])
                    ]),
                    
]),
html.Div([
            html.Img(src='/assets/re.gif', style={'position': 'absolute', 'margin-bottom': '100px', 'left': '600px', 'width': '700px', 'height': '700px', 'z-index': '4'})
        ]),
        html.Div([
        
        
        # html.H2('The subject of this study revolves around the realm of energy, specifically focusing on renewable energy sources and the process of <span style="color: red;">biomass gasification</span> for power production. The data under scrutiny is sourced from a downdraft biomass gasification-power production plant, encompassing a diverse range of biomass materials and varying operational parameters. This data is organized and presented in the form of a table, serving as a crucial resource for analyzing the performance and efficiency of <span style="color: blue;">biomass gasification</span> as a sustainable energy generation method across different feedstock types and operational settings.', 
        #         style={'color': '#3D3635','display':'center','text-align': 'justify ','font-size':'20px'}),
                
    dcc.Markdown('''
 Welcome to our innovative Smart Energy Management System designed specifically for agricultural applications. At **Smart Green**, we recognized the critical role agriculture plays in our lives and the challenges faced by farmers and growers in optimizing energy consumption. 
                 

With a deep commitment to sustainability and the environment, we developed a cutting-edge solution that harnesses the power of renewable energy sources to not only reduce operational costs but also minimize the ecological footprint of agriculture. 
                 
                 
Our Smart Energy Management System integrates **solar**, **biomass** and other renewable sources to provide a reliable and eco-friendly energy supply for farms. By doing so, we empower agricultural businesses to embrace the future of farming, **where sustainability meets efficiency**, and environmental responsibility aligns seamlessly with economic growth. 
                 

    ''', dangerously_allow_html=True, style={'color': 'black', 'text-align': 'justify','font-size':'20px'})
    ],  style={'margin-top':'700px', 'margin-left':'300px','text-align': 'center',
  'border-radius': '10px','height':'600px',
  'width':'1400px', 'padding': '60px', 'margin-bottom':'50px'}  # 'border': '2px solid black',Apply className to the div wrapping the pie chart
),

html.Div([
            html.Img(src='/assets/cow.gif', style={'top':'800px','position': 'absolute', 'margin-top': '40px', 
                                                   'right': '150px', 'width': '150px', 'height': '150px', 'z-index': '4'})
        ]),
html.Div([
            html.Img(src='/assets/cow2.gif', style={'top':'1300px','position': 'absolute', 'margin-top': '40px',
                                                     'right': '900px', 'width': '150px', 'height': '150px', 'z-index': '4'})
        ]),
        html.Div([
            html.Img(src='/assets/cow3.gif', style={'top':'1800px','position': 'absolute', 'margin-top': '40px',
                                                     'right': '300px', 'width': '150px', 'height': '150px', 'z-index': '4'})
        ]),
        
html.Div([
            html.Img(src='/assets/farmm.gif', style={'top':'1500px','position': 'absolute', 'margin-top': '40px',
                                                      'left': '180px', 'width': '450px', 'height': '450px', 'z-index': '4'})
        ]),

html.Div([
        
        
        # html.H2('The subject of this study revolves around the realm of energy, specifically focusing on renewable energy sources and the process of <span style="color: red;">biomass gasification</span> for power production. The data under scrutiny is sourced from a downdraft biomass gasification-power production plant, encompassing a diverse range of biomass materials and varying operational parameters. This data is organized and presented in the form of a table, serving as a crucial resource for analyzing the performance and efficiency of <span style="color: blue;">biomass gasification</span> as a sustainable energy generation method across different feedstock types and operational settings.', 
        #         style={'color': '#3D3635','display':'center','text-align': 'justify ','font-size':'20px'}),
                
    dcc.Markdown('''
                 
This project was executed through a collaborative effort involving both the [LaRINA](https://www.linkedin.com/company/larina-ucar/?originalSubdomain=tn) lab in Tunisia and [InES](https://www.thi.de/forschung/institut-fuer-neue-energie-systeme-ines/) institute in Germany, bringing together the expertise of two renowned institutions to create a solution that sets new standards for sustainable agricultural energy management. Join us in revolutionizing the agricultural industry and embrace a greener, more sustainable future for your farm.
                 

    ''', dangerously_allow_html=True, style={'color': 'black', 'text-align': 'justify','font-size':'20px'})
    ],  style={'margin-top':'100px', 'margin-left':'500px','text-align': 'center',
  'border-radius': '10px','height':'600px',
  'width':'1200px', 'padding': '60px', 'margin-bottom':'50px'}  # 'border': '2px solid black',Apply className to the div wrapping the pie chart
),
])