# -*- coding: utf-8 -*-
"""Tablero 3.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/17xq9DUXpYVAZpbmy8CDMC7QTU--_oahb
"""

pip install dash

pip install dash-bootstrap-components

pip install joblib

import dash
from dash import html, dcc, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import joblib  # Para cargar el modelo
import numpy as np
import plotly.express as px

url = 'https://github.com/jsperdomobe/proyecto/blob/main/Datos/train.csv?raw=true'
df = pd.read_csv(url, index_col=0)

df

df['Gender'].unique()

df['Age'].unique()

df['Academic Pressure'].unique()

# Boxplot de presión académica por género
fig_academic_pressure_gender = px.box(df, x='Gender', y='Academic Pressure', title='Presión Académica por Género')
fig_academic_pressure_gender.update_traces(line_color='blue')
fig_academic_pressure_gender.update_layout(paper_bgcolor='rgba(0, 0, 255, 0)', plot_bgcolor='rgba(0, 0, 255,0.1 )', font_color='dark blue')

fig_sleep_hours = px.bar(df, x='Sleep Duration',y='Depression', title='Distribución de Horas de Sueño')
#fig_sleep_hours.update_traces(line_color='white')
fig_sleep_hours.update_layout(
            paper_bgcolor='rgba(0, 0, 0, 0.5)',  # Color de fondo translúcido
            plot_bgcolor='rgba(0, 0, 0, 0)',    # Color de fondo del gráfico transparente
            font_color='white' )

df['Academic Pressure'].unique()

fig_academic_pressure = px.box(df, x='Gender', y='Work Pressure', title='Distribución de Presión trabajadora')
fig_academic_pressure.update_traces(line_color='white')
fig_academic_pressure.update_layout(
            paper_bgcolor='rgba(0, 0, 0, 0.5)',  # Color de fondo translúcido
            plot_bgcolor='rgba(0, 0, 0, 0)',    # Color de fondo del gráfico transparente
            font_color='white'        )

fig = px.pie(df, values='pop', names='country', title='Population of European continent')



###############################33
import dash
from dash import html, dcc, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import joblib  # Para cargar el modelo
import numpy as np
import plotly.express as px

# Inicializar la app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Cargar el modelo preentrenado
# modelo = joblib.load('modelo_entrenado.pkl')  # Asegúrate de que el nombre coincida con tu archivo

# Cargar el DataFrame desde la URL proporcionada
url = 'https://github.com/jsperdomobe/proyecto/blob/main/Datos/train.csv?raw=true'
df = pd.read_csv(url, index_col=0)

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
                                    {'label': 'Masculino', 'value': 'Male'},
                                    {'label': 'Femenino', 'value': 'Female'},
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

                    html.Div(id='output-div', style={'color': 'white', 'marginTop': '150px'})
                ])
               ])

            ]),
        dcc.Tab(label='Análisis Estadístico', children=[
            html.Div(style={'backgroundImage': 'url("https://raw.githubusercontent.com/jsperdomobe/proyecto/main/Imagen/yy123.jpg")',
                            'height': '100vh',
                            'backgroundSize': 'cover',
                            'padding': '130px'},
                     children=[
                dbc.Button("Actualizar Gráficas", id="update-button", n_clicks=0, style={'marginBottom': '20px'}),
                dbc.Row([
                    dbc.Col(dcc.Graph(id='histogram-academic-pressure'), width=6),
                    dbc.Col(dcc.Graph(id='histogram-work-pressure'), width=6),
                ]),
                dbc.Row([
                    dbc.Col(dcc.Graph(id='histogram-sleep-hours'), width=6),
                    dbc.Col(dcc.Graph(id='boxplot-academic-pressure-gender'), width=6),
                ]),
            ])
        ])
    ])
])



# Callback para actualizar el análisis estadístico
@app.callback(
    Output('histogram-academic-pressure', 'figure'),
    Output('histogram-work-pressure', 'figure'),
    Output('histogram-sleep-hours', 'figure'),
    Output('boxplot-academic-pressure-gender', 'figure'),
    Input('update-button', 'n_clicks')  # Ahora el botón de actualización activa el callback
)
def update_graphs(n_clicks):
    if n_clicks > 0:
        # Gráfica de distribución de presión de trabajo
        fig_academic_pressure = px.box(df, x='Working Professional or Student', y='Depression', title='Distribución de depresion')
        #fig_academic_pressure.update_traces(line_color='white')
        fig_academic_pressure.update_layout(
            paper_bgcolor='rgba(0, 0, 0, 0.5)',  # Color de fondo translúcido
            plot_bgcolor='rgba(0, 0, 0, 0)',    # Color de fondo del gráfico transparente
            font_color='white'        )

        # Gráfica de distribución de presión laboral
        fig_work_pressure = px.box(df, x='Gender', y='Work Pressure', title='Distribución de Presión Laboral')
        #fig_work_pressure.update_traces(line_color='white')
        fig_work_pressure.update_layout(
            paper_bgcolor='rgba(0, 0, 0, 0.5)',  # Color de fondo translúcido
            plot_bgcolor='rgba(0, 0, 0, 0)',    # Color de fondo del gráfico transparente
            font_color='white'
        )

        # Gráfica de distribución de horas de sueño
        fig_sleep_hours = px.bar(df, x='Sleep Duration',y='Depression', title='Distribución de Horas de Sueño')
        #fig_sleep_hours.update_traces(line_color='white')
        fig_sleep_hours.update_layout(
            paper_bgcolor='rgba(0, 0, 0, 0.5)',  # Color de fondo translúcido
            plot_bgcolor='rgba(0, 0, 0, 0)',    # Color de fondo del gráfico transparente
            font_color='white' )

        # Boxplot de presión académica por género
        fig_academic_pressure_gender = px.box(df, x='Gender', y='Academic Pressure', title='Presión Académica por Género')
        fig_academic_pressure_gender.update_traces(line_color='white')
        fig_academic_pressure_gender.update_layout(
            paper_bgcolor='rgba(0, 0, 0, 0.5)',  # Color de fondo translúcido
            plot_bgcolor='rgba(0, 0, 0, 0)',    # Color de fondo del gráfico transparente
            font_color='white')


    return fig_academic_pressure, fig_work_pressure, fig_sleep_hours, fig_academic_pressure_gender

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