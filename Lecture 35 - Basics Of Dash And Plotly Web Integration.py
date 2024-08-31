import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State

app = dash.Dash(__name__)

# Defining the layout
app.layout = html.Div([
    html.H1("ChatBot", style={'textAlign': 'center'}),
    dcc.Textarea(id='user-input', value='Ask something...', style={'width': '100%', 'height': 100}),
    html.Button('Submit', id='submit-button', n_clicks=0),
    html.Div(id='chatbot-output', style={'padding': '10px'})
])

# Creating a chatbot response
@app.callback(
    Output('chatbot-output', 'children'),
    [Input('submit-button', 'n_clicks')],
    [State('user-input', 'value')]
)
def update_output(n_clicks, user_input):
    if n_clicks > 0:
        return html.Div([
            html.P(f"You: {user_input}", style={'margin': '10px'}),
            html.P("Bot: I am training now, ask something else.", style={'margin': '10px', 'backgroundColor': 'beige', 'padding': '10px'})
        ])
    return "Ask me something!"

if __name__ == '__main__':
    app.run_server(debug=True)
