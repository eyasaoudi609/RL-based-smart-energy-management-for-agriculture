from dash import html


layout = html.Div([

    html.Div([
        html.Div([
            html.Div([
                html.Div([
                    html.H3('Eya Saoudi', style={"color": "#62a484",
                                                 'margin-left': '15px',"font-size": "40px",
                                                 'margin-top': '15px'}),
                    html.P('This project was jointly conducted by LaRINA Lab and InES institute in Germany and was presented by Eya Saoudi, a master student at ENSTAB, as her concluding thesis project.  '
                           ,
                           style={"color": "#073B3A",
                                  "font-size": "30px",
                                  'margin-left': '15px',
                                  'margin-right': '15px',
                                  'margin-top': '15px',
                                  'margin-bottom': '15px',
                                  'line-height': '1.2',
                                  'text-align': 'justify'
                                  }
                           ),
                    html.Div([
            
                        # html.A(href='https://community.plotly.com/u/jawabutt/summary', target='_blank',
                        #        children=[html.Img(src='/assets/plotly.ico', height="30px",
                        #                           style={"margin-top": '20px',
                        #                                  'margin-left': '15px',
                        #                                  'margin-bottom': '15px',
                        #                                  "background-color": "#35384b"})]),
                        html.A(href='https://www.linkedin.com/in/eyasaoudi/', target='_blank',
                               children=[html.Img(src='/assets/linkedin.png', height="30px",
                                                  style={"margin-top": '15px',
                                                         'margin-left': '15px',
                                                         'margin-bottom': '15px'})]),
                        html.A(href='mailto:saoudi_eya@yahoo.com', target='_blank',
                               children=[html.Img(src='/assets/mail.png', height="30px",
                                                  style={"margin-top": '20px',
                                                         'margin-left': '15px',
                                                         'margin-bottom': '15px',
                                                         "background-color": "#d3e8c9"})]),
])
                    ])
                ], className='result-box7',style={"margin-top": '200px','height':'800px',
                                                         'margin-left': '500px',
                                                         'margin-bottom': '15px',
                                                         "background-color": "#d3e8c9",
                                                         'width':'900px'}),
                html.Div([
            html.Img(src='/assets/prog.gif', style={'position': 'absolute', 'top': '500px', 'right': '700px', 'width': '450px', 'height': '450px', 'z-index': '4'})
        ]),
            ])
        ])
    ])
