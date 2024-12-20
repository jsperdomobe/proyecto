# -*- coding: utf-8 -*-
"""Tablero.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/12wbY_CSNSJGECX7pkuu5CizpqBspmZBo
"""

pip install dash

pip install dash-bootstrap-components

from dash import Dash, dcc, html
import dash_bootstrap_components as dbc

import plotly.graph_objects as go
import numpy as np
import random

app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])

fig = go.Figure(
    go.Scattergl(
        x = np.random.randn(1000),
        y = np.random.randn(1000),
        mode='markers',
        marker=dict(color=random.sample(['#ecf0f1']*500 + ["#3498db"]*500, 1000), line_width=1)
    )
)
fig.update_layout(plot_bgcolor='#010103', width=790, height=730,
                  xaxis_visible=False, yaxis_visible=False, showlegend=False, margin=dict(l=0,r=0,t=0,b=0))

app.layout = dbc.Container([
    html.Div([
        html.Div([
            html.H1([
                html.Span("Welcome"),
                html.Br(),
                html.Span("to my beautiful dashboard!")
            ]),
            html.
            P("This dashboard prototype shows how to create an effective layout."
              )
        ],
                 style={"vertical-alignment": "top", "height": 260}),
        html.Div([
            html.Div(
                dbc.RadioItems(
                    className='btn-group',
                    inputClassName='btn-check',
                    labelClassName="btn btn-outline-light",
                    labelCheckedClassName="btn btn-light",
                    options=[
                        {"label": "Graph", "value": 1},
                        {"label": "Table", "value": 2}
                    ],
                    value=1,
                    style={'width': '100%'}
                ), style={'width': 206}
            ),
            html.Div(
                dbc.Button(
                    "About",
                    className="btn btn-info",
                    n_clicks=0
                ), style={'width': 104})
        ], style={'margin-left': 15, 'margin-right': 15, 'display': 'flex'}),
        html.Div([
            html.Div([
                html.H2('Unclearable Dropdown:'),
                dcc.Dropdown(
                    options=[
                        {'label': 'Option A', 'value': 1},
                        {'label': 'Option B', 'value': 2},
                        {'label': 'Option C', 'value': 3}
                    ],
                    value=1,
                    clearable=False,
                    optionHeight=40,
                    className='customDropdown'
                )
            ]),
            html.Div([
                html.H2('Unclearable Dropdown:'),
                dcc.Dropdown(
                    options=[
                        {'label': 'Option A', 'value': 1},
                        {'label': 'Option B', 'value': 2},
                        {'label': 'Option C', 'value': 3}
                    ],
                    value=2,
                    clearable=False,
                    optionHeight=40,
                    className='customDropdown'
                )
            ]),
            html.Div([
                html.H2('Clearable Dropdown:'),
                dcc.Dropdown(
                    options=[
                        {'label': 'Option A', 'value': 1},
                        {'label': 'Option B', 'value': 2},
                        {'label': 'Option C', 'value': 3}
                    ],
                    clearable=True,
                    optionHeight=40,
                    className='customDropdown'
                )
            ])
        ], style={'margin-left': 15, 'margin-right': 15, 'margin-top': 30}),
        html.Div(
            html.Img(src='assets/image.svg',
                     style={'margin-left': 15, 'margin-right': 15, 'margin-top': 30, 'width': 310})
        )
    ], style={
        'width': 340,
        'margin-left': 35,
        'margin-top': 35,
        'margin-bottom': 35
    }),
    html.Div(
        [
            html.Div(
                dcc.Graph(
                    figure=fig
                ),
                     style={'width': 790}),
            html.Div([
                html.H2('Output 1:'),
                html.Div(className='Output'),
                html.H2('Output 2:'),
                html.Div(html.H3("Selected Value"), className='Output')
            ], style={'width': 198})
        ],
        style={
            'width': 990,
            'margin-top': 35,
            'margin-right': 35,
            'margin-bottom': 35,
            'display': 'flex'
        })
],
                           fluid=True,
                           style={'display': 'flex'},
                           className='dashboard-container')
if __name__ == "__main__":
    app.run_server(debug=True)

import dash
from dash import html, dcc
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc

# Initialize the app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H2("Alertas tempranas"), width=8),
    ], align="center"),

    dbc.Row([
        dbc.Col([
            dbc.Label("Genero"),
            dcc.Dropdown(
                id='gender-dropdown',
                options=[
                    {'label': 'Masculino', 'value': 'M'},
                    {'label': 'Femenino', 'value': 'F'},
                    {'label': 'Otro', 'value': 'O'}
                ],
                placeholder="Select Gender"
            ),
            dbc.Label("Edad"),
            dcc.Dropdown(
                id='age-dropdown',
                options=[
                    {'label': '18-25', 'value': '18-25'},
                    {'label': '26-35', 'value': '26-35'},
                    {'label': '36-45', 'value': '36-45'},
                ],
                placeholder="Select Age"
            ),
            dbc.Label("Profesion"),
            dcc.RadioItems(
                options=[
                    {'label': 'Estudiante', 'value': 'student'},
                    {'label': 'Trabajador', 'value': 'worker'},
                    {'label': 'Otro', 'value': 'other'}
                ],
                inline=True
            ),
        ], width=4),

        dbc.Col([
            dbc.Label("Presión académica"),
            dcc.Slider(id='academic-pressure-slider', min=0, max=10, step=1),

            dbc.Label("Presión laboral"),
            dcc.Slider(id='work-pressure-slider', min=0, max=10, step=1),

            dbc.Label("Horas de sueño"),
            dcc.Dropdown(
                id='sleep-dropdown',
                options=[
                    {'label': 'Menos de 4 horas', 'value': '<4'},
                    {'label': '4-6 horas', 'value': '4-6'},
                    {'label': '6-8 horas', 'value': '6-8'},
                    {'label': 'Más de 8 horas', 'value': '>8'}
                ],
                placeholder="Select Sleep Hours"
            ),
        ], width=4),

        dbc.Col([
            dbc.Label("Pensamiento suicidas"),
            dbc.Checklist(
                options=[{'label': "SI", 'value': 'YES'},
                         {'label': "NO", 'value': 'NO'}],
                inline=True
            ),
        ], width=4)
    ], align="start"),
])

if __name__ == '__main__':
    app.run_server(debug=True)

import dash
from dash import html, dcc, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd

# Initialize the app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Initialize a DataFrame to store responses
responses = pd.DataFrame(columns=["Gender", "Age", "Profession", "Academic Pressure", "Work Pressure", "Sleep Hours", "Suicidal Thoughts"])

app.layout = dbc.Container([
    html.Div(style={'backgroundImage': 'url("/assets/XX123.jpg")', 'height': '100vh', 'backgroundSize': 'cover'}),

    dbc.Row([
        dbc.Col(html.H2("Alertas tempranas"), width=8),
    ], align="center"),

    dbc.Row([
        dbc.Col([
            dbc.Label("Genero"),
            dcc.Dropdown(
                id='gender-dropdown',
                options=[
                    {'label': 'Masculino', 'value': 'M'},
                    {'label': 'Femenino', 'value': 'F'},
                    {'label': 'Otro', 'value': 'O'}
                ],
                placeholder="Select Gender"
            ),
            dbc.Label("Edad"),
            dcc.Dropdown(
                id='age-dropdown',
                options=[
                    {'label': '18-25', 'value': '18-25'},
                    {'label': '26-35', 'value': '26-35'},
                    {'label': '36-45', 'value': '36-45'},
                ],
                placeholder="Select Age"
            ),
            dbc.Label("Profesion"),
            dcc.RadioItems(
                id='profession-radio',
                options=[
                    {'label': 'Estudiante', 'value': 'student'},
                    {'label': 'Trabajador', 'value': 'worker'},
                    {'label': 'Otro', 'value': 'other'}
                ],
                inline=True
            ),
        ], width=4),

        dbc.Col([
            dbc.Label("Presión académica"),
            dcc.Slider(id='academic-pressure-slider', min=0, max=10, step=1, value=0),

            dbc.Label("Presión laboral"),
            dcc.Slider(id='work-pressure-slider', min=0, max=10, step=1, value=0),

            dbc.Label("Horas de sueño"),
            dcc.Dropdown(
                id='sleep-dropdown',
                options=[
                    {'label': 'Menos de 4 horas', 'value': '<4'},
                    {'label': '4-6 horas', 'value': '4-6'},
                    {'label': '6-8 horas', 'value': '6-8'},
                    {'label': 'Más de 8 horas', 'value': '>8'}
                ],
                placeholder="Select Sleep Hours"
            ),
        ], width=4),

        dbc.Col([
            dbc.Label("Pensamiento suicidas"),
            dcc.Checklist(
                id='suicidal-thoughts-checklist',
                options=[{'label': "SI", 'value': 'YES'},
                         {'label': "NO", 'value': 'NO'}],
                inline=True
            ),
            dbc.Button("Guardar Respuestas", id='submit-button', n_clicks=0)
        ], width=4)
    ], align="start"),

    html.Div(id='output-div')
])

@app.callback(
    Output('output-div', 'children'),
    Input('submit-button', 'n_clicks'),
    Input('gender-dropdown', 'value'),
    Input('age-dropdown', 'value'),
    Input('profession-radio', 'value'),
    Input('academic-pressure-slider', 'value'),
    Input('work-pressure-slider', 'value'),
    Input('sleep-dropdown', 'value'),
    Input('suicidal-thoughts-checklist', 'value')
)
def update_output(n_clicks, gender, age, profession, academic_pressure, work_pressure, sleep_hours, suicidal_thoughts):
    global responses

    if n_clicks > 0:
        # Store the responses in the DataFrame
        responses = responses.append({
            "Gender": gender,
            "Age": age,
            "Profession": profession,
            "Academic Pressure": academic_pressure,
            "Work Pressure": work_pressure,
            "Sleep Hours": sleep_hours,
            "Suicidal Thoughts": suicidal_thoughts
        }, ignore_index=True)

        return f"Respuestas guardadas:\n{responses.tail(1).to_dict(orient='records')[0]}"

    return "Presione 'Guardar Respuestas' para guardar sus respuestas."

if __name__ == '__main__':
    app.run_server(debug=True)

import dash
from dash import html, dcc, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd

# Initialize the app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Initialize a DataFrame to store responses
responses = pd.DataFrame(columns=["Gender", "Age", "Profession", "Academic Pressure", "Work Pressure", "Sleep Hours", "Suicidal Thoughts"])

app.layout = dbc.Container([
    html.Div(style={'backgroundImage': 'url("XX123.jpg")', 'height': '100vh', 'backgroundSize': 'cover'}),

    dbc.Row([
        dbc.Col(html.H2("Alertas tempranas"), width=8),
    ], align="center"),

    dbc.Row([
        dbc.Col([
            dbc.Label("Genero"),
            dcc.Dropdown(
                id='gender-dropdown',
                options=[
                    {'label': 'Masculino', 'value': 'M'},
                    {'label': 'Femenino', 'value': 'F'},
                    {'label': 'Otro', 'value': 'O'}
                ],
                placeholder="Select Gender"
            ),
            dbc.Label("Edad"),
            dcc.Dropdown(
                id='age-dropdown',
                options=[
                    {'label': '18-25', 'value': '18-25'},
                    {'label': '26-35', 'value': '26-35'},
                    {'label': '36-45', 'value': '36-45'},
                ],
                placeholder="Select Age"
            ),
            dbc.Label("Profesion"),
            dcc.RadioItems(
                id='profession-radio',
                options=[
                    {'label': 'Estudiante', 'value': 'student'},
                    {'label': 'Trabajador', 'value': 'worker'},
                    {'label': 'Otro', 'value': 'other'}
                ],
                inline=True
            ),
        ], width=4),

        dbc.Col([
            dbc.Label("Presión académica"),
            dcc.Slider(id='academic-pressure-slider', min=0, max=10, step=1, value=0),

            dbc.Label("Presión laboral"),
            dcc.Slider(id='work-pressure-slider', min=0, max=10, step=1, value=0),

            dbc.Label("Horas de sueño"),
            dcc.Dropdown(
                id='sleep-dropdown',
                options=[
                    {'label': 'Menos de 4 horas', 'value': '<4'},
                    {'label': '4-6 horas', 'value': '4-6'},
                    {'label': '6-8 horas', 'value': '6-8'},
                    {'label': 'Más de 8 horas', 'value': '>8'}
                ],
                placeholder="Select Sleep Hours"
            ),
        ], width=4),

        dbc.Col([
            dbc.Label("Pensamiento suicidas"),
            dcc.Checklist(
                id='suicidal-thoughts-checklist',
                options=[{'label': "SI", 'value': 'YES'},
                         {'label': "NO", 'value': 'NO'}],
                inline=True
            ),
            dbc.Button("Guardar Respuestas", id='submit-button', n_clicks=0)
        ], width=4)
    ], align="start"),

    html.Div(id='output-div')
])

@app.callback(
    Output('output-div', 'children'),
    Input('submit-button', 'n_clicks'),
    Input('gender-dropdown', 'value'),
    Input('age-dropdown', 'value'),
    Input('profession-radio', 'value'),
    Input('academic-pressure-slider', 'value'),
    Input('work-pressure-slider', 'value'),
    Input('sleep-dropdown', 'value'),
    Input('suicidal-thoughts-checklist', 'value')
)
def update_output(n_clicks, gender, age, profession, academic_pressure, work_pressure, sleep_hours, suicidal_thoughts):
    global responses

    if n_clicks > 0:
        # Store the responses in the DataFrame
        responses = responses.append({
            "Gender": gender,
            "Age": age,
            "Profession": profession,
            "Academic Pressure": academic_pressure,
            "Work Pressure": work_pressure,
            "Sleep Hours": sleep_hours,
            "Suicidal Thoughts": suicidal_thoughts
        }, ignore_index=True)

        return f"Respuestas guardadas:\n{responses.tail(1).to_dict(orient='records')[0]}"

    return "Presione 'Guardar Respuestas' para guardar sus respuestas."

if __name__ == '__main__':
    app.run_server(debug=True)

import dash
from dash import html, dcc, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd

# Inicializar la app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Inicializar un DataFrame para almacenar respuestas
responses = pd.DataFrame(columns=["Gender", "Age", "Profession", "Academic Pressure", "Work Pressure", "Sleep Hours", "Suicidal Thoughts"])

app.layout = dbc.Container([
    dcc.Tabs([
        dcc.Tab(label='Tablero', children=[
            html.Div(style={'backgroundImage': 'XX123.jpg', 'height': '100vh', 'backgroundSize': 'cover'}),
            dbc.Row([
                dbc.Col(html.H2("Alertas tempranas"), width=8),
            ], align="center"),
            dbc.Row([
                dbc.Col([
                    dbc.Label("Género"),
                    dcc.Dropdown(
                        id='gender-dropdown',
                        options=[
                            {'label': 'Masculino', 'value': 'M'},
                            {'label': 'Femenino', 'value': 'F'},
                            {'label': 'Otro', 'value': 'O'}
                        ],
                        placeholder="Selecciona Género"
                    ),
                    dbc.Label("Edad"),
                    dcc.Dropdown(
                        id='age-dropdown',
                        options=[
                            {'label': '18-25', 'value': '18-25'},
                            {'label': '26-35', 'value': '26-35'},
                            {'label': '36-45', 'value': '36-45'},
                        ],
                        placeholder="Selecciona Edad"
                    ),
                    dbc.Label("Profesión"),
                    dcc.RadioItems(
                        id='profession-radio',
                        options=[
                            {'label': 'Estudiante', 'value': 'student'},
                            {'label': 'Trabajador', 'value': 'worker'},
                            {'label': 'Otro', 'value': 'other'}
                        ],
                        inline=True
                    ),
                ], width=4),

                dbc.Col([
                    dbc.Label("Presión académica"),
                    dcc.Slider(id='academic-pressure-slider', min=0, max=10, step=1, value=0),

                    dbc.Label("Presión laboral"),
                    dcc.Slider(id='work-pressure-slider', min=0, max=10, step=1, value=0),

                    dbc.Label("Horas de sueño"),
                    dcc.Dropdown(
                        id='sleep-dropdown',
                        options=[
                            {'label': 'Menos de 4 horas', 'value': '<4'},
                            {'label': '4-6 horas', 'value': '4-6'},
                            {'label': '6-8 horas', 'value': '6-8'},
                            {'label': 'Más de 8 horas', 'value': '>8'}
                        ],
                        placeholder="Selecciona Horas de Sueño"
                    ),
                ], width=4),

                dbc.Col([
                    dbc.Label("Pensamientos suicidas"),
                    dcc.Checklist(
                        id='suicidal-thoughts-checklist',
                        options=[{'label': "SI", 'value': 'YES'},
                                 {'label': "NO", 'value': 'NO'}],
                        inline=True
                    ),
                    dbc.Button("Guardar Respuestas", id='submit-button', n_clicks=0)
                ], width=4)
            ], align="start"),

            html.Div(id='output-div')
        ]),
        dcc.Tab(label='Análisis Estadístico', children=[
            html.Div(style={'backgroundImage': 'url("yy12.jpg")', 'height': '100vh', 'backgroundSize': 'cover', 'padding': '20px'}, children=[
                html.H2("Análisis Estadístico"),
                html.Div(id='stats-output')
            ])
        ])
    ])
])

@app.callback(
    Output('output-div', 'children'),
    Input('submit-button', 'n_clicks'),
    Input('gender-dropdown', 'value'),
    Input('age-dropdown', 'value'),
    Input('profession-radio', 'value'),
    Input('academic-pressure-slider', 'value'),
    Input('work-pressure-slider', 'value'),
    Input('sleep-dropdown', 'value'),
    Input('suicidal-thoughts-checklist', 'value')
)
def update_output(n_clicks, gender, age, profession, academic_pressure, work_pressure, sleep_hours, suicidal_thoughts):
    global responses

    if n_clicks > 0:
        # Store the responses in the DataFrame
        responses = responses.append({
            "Gender": gender,
            "Age": age,
            "Profession": profession,
            "Academic Pressure": academic_pressure,
            "Work Pressure": work_pressure,
            "Sleep Hours": sleep_hours,
            "Suicidal Thoughts": suicidal_thoughts
        }, ignore_index=True)

        return f"Respuestas guardadas:\n{responses.tail(1).to_dict(orient='records')[0]}"

    return "Presione 'Guardar Respuestas' para guardar sus respuestas."

if __name__ == '__main__':
    app.run_server(debug=True)

import dash
from dash import html, dcc, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import joblib  # Para cargar el modelo
import numpy as np

# Inicializar la app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Cargar el modelo preentrenado
# modelo = joblib.load('modelo_entrenado.pkl')  # Asegúrate de que el nombre coincida con tu archivo

# Inicializar un DataFrame para almacenar respuestas
responses = pd.DataFrame(columns=["Gender", "Age", "Profession", "Academic Pressure", "Work Pressure", "Sleep Hours", "Suicidal Thoughts"])

app.layout = dbc.Container([
    dcc.Tabs([
        dcc.Tab(label='Tablero', children=[
            html.Div(style={
                'backgroundImage': 'url("https://raw.githubusercontent.com/jsperdomobe/proyecto/main/Imagen/XX123.jpg")',
                'height': '90vh',
                'backgroundSize': 'cover'
            }, children=[
                html.Div(style={'backgroundColor': 'rgba(0, 0, 0, 0.5)', 'height': '100%', 'padding': '180px'}, children=[
                    dbc.Row([
                        dbc.Col(html.H2("Alertas tempranas", style={'color': 'white', 'textAlign': 'center'}), width=8),
                    ], align="center"),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Género", style={'color': 'white'}),
                            dcc.Dropdown(
                                id='gender-dropdown',
                                options=[
                                    {'label': 'Masculino', 'value': 'M'},
                                    {'label': 'Femenino', 'value': 'F'},
                                    {'label': 'Otro', 'value': 'O'}
                                ],
                                placeholder="Selecciona Género"
                            ),
                            dbc.Label("Edad", style={'color': 'white'}),
                            dcc.Dropdown(
                                id='age-dropdown',
                                options=[
                                    {'label': '18-25', 'value': '18-25'},
                                    {'label': '26-35', 'value': '26-35'},
                                    {'label': '36-45', 'value': '36-45'},
                                ],
                                placeholder="Selecciona Edad"
                            ),
                            dbc.Label("Profesión", style={'color': 'white'}),
                            dcc.RadioItems(
                                id='profession-radio',
                                options=[
                                    {'label': 'Estudiante', 'value': 'student'},
                                    {'label': 'Trabajador', 'value': 'worker'},
                                    {'label': 'Otro', 'value': 'other'}
                                ],
                                inline=True,
                                style={'color': 'white'}
                            ),
                        ], width=4),

                        dbc.Col([
                            dbc.Label("Presión académica", style={'color': 'white'}),
                            dcc.Slider(id='academic-pressure-slider', min=0, max=10, step=1, value=0,
                                       marks={i: str(i) for i in range(11)}),

                            dbc.Label("Presión laboral", style={'color': 'white'}),
                            dcc.Slider(id='work-pressure-slider', min=0, max=10, step=1, value=0,
                                       marks={i: str(i) for i in range(11)}),

                            dbc.Label("Horas de sueño", style={'color': 'white'}),
                            dcc.Dropdown(
                                id='sleep-dropdown',
                                options=[
                                    {'label': 'Menos de 4 horas', 'value': '<4'},
                                    {'label': '4-6 horas', 'value': '4-6'},
                                    {'label': '6-8 horas', 'value': '6-8'},
                                    {'label': 'Más de 8 horas', 'value': '>8'}
                                ],
                                placeholder="Selecciona Horas de Sueño"
                            ),
                        ], width=4),

                        dbc.Col([
                            dbc.Label("Pensamientos suicidas", style={'color': 'white'}),
                            dcc.Checklist(
                                id='suicidal-thoughts-checklist',
                                options=[{'label': "SI", 'value': 'YES'},
                                         {'label': "NO", 'value': 'NO'}],
                                inline=True
                            ),
                            dbc.Button("Guardar Respuestas", id='submit-button', n_clicks=0)
                        ], width=4)
                    ], align="start"),

                    html.Div(id='output-div', style={'color': 'white', 'marginTop': '80px'})
                ])
               ])
              ])
            ]),
        dcc.Tab(label='Análisis Estadístico', children=[
            html.Div(style={'backgroundImage': 'url("https://raw.githubusercontent.com/jsperdomobe/proyecto/main/Imagen/yy12.jpg")',
                            'height': '90vh',
                            'backgroundSize': 'cover',
                            'padding': '20px'},
                     children=[
                html.H2("Análisis Estadístico"),
                html.Div(id='stats-output')
            ])
        ])
    ])

@app.callback(
    Output('output-div', 'children'),
    Input('submit-button', 'n_clicks'),
    Input('gender-dropdown', 'value'),
    Input('age-dropdown', 'value'),
    Input('profession-radio', 'value'),
    Input('academic-pressure-slider', 'value'),
    Input('work-pressure-slider', 'value'),
    Input('sleep-dropdown', 'value'),
    Input('suicidal-thoughts-checklist', 'value')
)
def update_output(n_clicks, gender, age, profession, academic_pressure, work_pressure, sleep_hours, suicidal_thoughts):
    global responses

    if n_clicks > 0:
        # Store the responses in the DataFrame
        responses = responses.append({
            "Gender": gender,
            "Age": age,
            "Profession": profession,
            "Academic Pressure": academic_pressure,
            "Work Pressure": work_pressure,
            "Sleep Hours": sleep_hours,
            "Suicidal Thoughts": suicidal_thoughts
        }, ignore_index=True)

        return f"Respuestas guardadas:\n{responses.tail(1).to_dict(orient='records')[0]}"

    return "Presione 'Guardar Respuestas' para guardar sus respuestas."

if __name__ == '__main__':
    app.run_server(debug=True)

import dash
from dash import html, dcc, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd

# Inicializar la app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Inicializar un DataFrame para almacenar respuestas
responses = pd.DataFrame(columns=["Gender", "Age", "Profession", "Academic Pressure", "Work Pressure", "Sleep Hours", "Suicidal Thoughts"])

app.layout = dbc.Container([
    dcc.Tabs([
        dcc.Tab(label='Tablero', children=[
            html.Div(style={'backgroundImage': 'url("XX123.jpg")', 'height': '100vh', 'backgroundSize': 'cover'}, children=[
                dbc.Row([
                    dbc.Col(html.H2("Alertas tempranas"), width=8),
                ], align="center"),
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Género"),
                        dcc.Dropdown(
                            id='gender-dropdown',
                            options=[
                                {'label': 'Masculino', 'value': 'M'},
                                {'label': 'Femenino', 'value': 'F'},
                                {'label': 'Otro', 'value': 'O'}
                            ],
                            placeholder="Selecciona Género"
                        ),
                        dbc.Label("Edad"),
                        dcc.Dropdown(
                            id='age-dropdown',
                            options=[
                                {'label': '18-25', 'value': '18-25'},
                                {'label': '26-35', 'value': '26-35'},
                                {'label': '36-45', 'value': '36-45'},
                            ],
                            placeholder="Selecciona Edad"
                        ),
                        dbc.Label("Profesión"),
                        dcc.RadioItems(
                            id='profession-radio',
                            options=[
                                {'label': 'Estudiante', 'value': 'student'},
                                {'label': 'Trabajador', 'value': 'worker'},
                                {'label': 'Otro', 'value': 'other'}
                            ],
                            inline=True
                        ),
                    ], width=4),

                    dbc.Col([
                        dbc.Label("Presión académica"),
                        dcc.Slider(id='academic-pressure-slider', min=0, max=10, step=1, value=0),

                        dbc.Label("Presión laboral"),
                        dcc.Slider(id='work-pressure-slider', min=0, max=10, step=1, value=0),

                        dbc.Label("Horas de sueño"),
                        dcc.Dropdown(
                            id='sleep-dropdown',
                            options=[
                                {'label': 'Menos de 4 horas', 'value': '<4'},
                                {'label': '4-6 horas', 'value': '4-6'},
                                {'label': '6-8 horas', 'value': '6-8'},
                                {'label': 'Más de 8 horas', 'value': '>8'}
                            ],
                            placeholder="Selecciona Horas de Sueño"
                        ),
                    ], width=4),

                    dbc.Col([
                        dbc.Label("Pensamientos suicidas"),
                        dcc.Checklist(
                            id='suicidal-thoughts-checklist',
                            options=[{'label': "SI", 'value': 'YES'},
                                     {'label': "NO", 'value': 'NO'}],
                            inline=True
                        ),
                        dbc.Button("Guardar Respuestas", id='submit-button', n_clicks=0)
                    ], width=4)
                ], align="start"),

                html.Div(id='output-div')
            ])
        ]),
        dcc.Tab(label='Análisis Estadístico', children=[
            html.Div(style={'backgroundImage': 'url("yy12.jpg")', 'height': '100vh', 'backgroundSize': 'cover', 'padding': '20px'}, children=[
                html.H2("Análisis Estadístico"),
                html.Div(id='stats-output')
            ])
        ])
    ])
])

@app.callback(
    Output('output-div', 'children'),
    Input('submit-button', 'n_clicks'),
    Input('gender-dropdown', 'value'),
    Input('age-dropdown', 'value'),
    Input('profession-radio', 'value'),
    Input('academic-pressure-slider', 'value'),
    Input('work-pressure-slider', 'value'),
    Input('sleep-dropdown', 'value'),
    Input('suicidal-thoughts-checklist', 'value')
)
def update_output(n_clicks, gender, age, profession, academic_pressure, work_pressure, sleep_hours, suicidal_thoughts):
    global responses

    if n_clicks > 0:
        # Store the responses in the DataFrame
        responses = responses.append({
            "Gender": gender,
            "Age": age,
            "Profession": profession,
            "Academic Pressure": academic_pressure,
            "Work Pressure": work_pressure,
            "Sleep Hours": sleep_hours,
            "Suicidal Thoughts": suicidal_thoughts
        }, ignore_index=True)

        return f"Respuestas guardadas:\n{responses.tail(1).to_dict(orient='records')[0]}"

    return "Presione 'Guardar Respuestas' para guardar sus respuestas."

if __name__ == '__main__':
    app.run_server(debug=True)

###############################33
import dash
from dash import html, dcc, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import joblib  # Para cargar el modelo
import numpy as np

# Inicializar la app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Cargar el modelo preentrenado
# modelo = joblib.load('modelo_entrenado.pkl')  # Asegúrate de que el nombre coincida con tu archivo

# Inicializar un DataFrame para almacenar respuestas
responses = pd.DataFrame(columns=["Gender", "Age", "Profession", "Academic Pressure", "Work Pressure", "Sleep Hours", "Suicidal Thoughts"])

app.layout = dbc.Container([
    dcc.Tabs([
        dcc.Tab(label='Tablero', children=[
            html.Div(style={
                'backgroundImage': 'url("https://raw.githubusercontent.com/jsperdomobe/proyecto/main/Imagen/XX123.jpg")',
                'height': '80vh',
                'backgroundSize': 'cover'
            }, children=[
                html.Div(style={'backgroundColor': 'rgba(0, 0, 0, 0.5)', 'height': '100%', 'padding': '180px'}, children=[
                    dbc.Row([
                        dbc.Col(html.H2("Alertas tempranas", style={'color': 'white', 'textAlign': 'center'}), width=8),
                    ], align="center"),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Género", style={'color': 'white'}),
                            dcc.Dropdown(
                                id='gender-dropdown',
                                options=[
                                    {'label': 'Masculino', 'value': 'M'},
                                    {'label': 'Femenino', 'value': 'F'},
                                    {'label': 'Otro', 'value': 'O'}
                                ],
                                placeholder="Selecciona Género"
                            ),
                            dbc.Label("Edad", style={'color': 'white'}),
                            dcc.Dropdown(
                                id='age-dropdown',
                                options=[
                                    {'label': '18-25', 'value': '18-25'},
                                    {'label': '26-35', 'value': '26-35'},
                                    {'label': '36-45', 'value': '36-45'},
                                ],
                                placeholder="Selecciona Edad"
                            ),
                            dbc.Label("Profesión", style={'color': 'white'}),
                            dcc.RadioItems(
                                id='profession-radio',
                                options=[
                                    {'label': 'Estudiante', 'value': 'student'},
                                    {'label': 'Trabajador', 'value': 'worker'},
                                    {'label': 'Otro', 'value': 'other'}
                                ],
                                inline=True,
                                style={'color': 'white'}
                            ),
                        ], width=4),

                        dbc.Col([
                            dbc.Label("Presión académica", style={'color': 'white'}),
                            dcc.Slider(id='academic-pressure-slider', min=0, max=10, step=1, value=0,
                                       marks={i: str(i) for i in range(11)}),

                            dbc.Label("Presión laboral", style={'color': 'white'}),
                            dcc.Slider(id='work-pressure-slider', min=0, max=10, step=1, value=0,
                                       marks={i: str(i) for i in range(11)}),

                            dbc.Label("Horas de sueño", style={'color': 'white'}),
                            dcc.Dropdown(
                                id='sleep-dropdown',
                                options=[
                                    {'label': 'Menos de 4 horas', 'value': '<4'},
                                    {'label': '4-6 horas', 'value': '4-6'},
                                    {'label': '6-8 horas', 'value': '6-8'},
                                    {'label': 'Más de 8 horas', 'value': '>8'}
                                ],
                                placeholder="Selecciona Horas de Sueño"
                            ),
                        ], width=4),

                        dbc.Col([
                            dbc.Label("Pensamientos suicidas", style={'color': 'white'}),
                            dcc.Checklist(
                                id='suicidal-thoughts-checklist',
                                options=[{'label': "SI", 'value': 'YES'},
                                         {'label': "NO", 'value': 'NO'}],
                                inline=True
                            ),
                            dbc.Button("Guardar Respuestas", id='submit-button', n_clicks=0)
                        ], width=4)
                    ], align="start"),

                    html.Div(id='output-div', style={'color': 'white', 'marginTop': '80px'})
                ])
               ])

            ]),
        dcc.Tab(label='Análisis Estadístico', children=[
            html.Div(style={'backgroundImage': 'url("https://raw.githubusercontent.com/jsperdomobe/proyecto/main/Imagen/yy12.jpg")',
                            'height': '80vh',
                            'backgroundSize': 'cover',
                            'padding': '20px'},
                     children=[
                html.H2("Análisis Estadístico"),
                html.Div(id='stats-output')
            ])
        ])
    ])
  ])

@app.callback(
    Output('output-div', 'children'),
    Input('submit-button', 'n_clicks'),
    Input('gender-dropdown', 'value'),
    Input('age-dropdown', 'value'),
    Input('profession-radio', 'value'),
    Input('academic-pressure-slider', 'value'),
    Input('work-pressure-slider', 'value'),
    Input('sleep-dropdown', 'value'),
    Input('suicidal-thoughts-checklist', 'value')
)
def update_output(n_clicks, gender, age, profession, academic_pressure, work_pressure, sleep_hours, suicidal_thoughts):
    global responses

    if n_clicks > 0:
        # Store the responses in the DataFrame
        responses = responses.append({
            "Gender": gender,
            "Age": age,
            "Profession": profession,
            "Academic Pressure": academic_pressure,
            "Work Pressure": work_pressure,
            "Sleep Hours": sleep_hours,
            "Suicidal Thoughts": suicidal_thoughts
        }, ignore_index=True)

        return f"Respuestas guardadas:\n{responses.tail(1).to_dict(orient='records')[0]}"

    return "Presione 'Guardar Respuestas' para guardar sus respuestas."

if __name__ == '__main__':
    app.run_server(debug=True)