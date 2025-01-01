import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import joblib
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
import re
import plotly.graph_objs as go

# Load pre-trained model and vectorizer
model = joblib.load('news_classifier_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Set up NLTK tools
tokenizer = RegexpTokenizer(r'\w+')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

def preprocess_text(text):
    """Preprocess text for prediction (tokenization, stopword removal, stemming)."""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters
    tokens = tokenizer.tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    tokens = [stemmer.stem(word) for word in tokens]  # Apply stemming
    return ' '.join(tokens)

# Create Dash app
app = dash.Dash(__name__)

# Initialize an empty dictionary to keep track of predictions
prediction_counts = {
    'Sports': 0,
    'Tech': 0,
    'Business': 0,
    'Health': 0,
    'Entertainment': 0,
    'Politics': 0,
    'World News': 0,
    'Lifestyle': 0
}

# App layout
app.layout = html.Div(style={
    'backgroundColor': '#f8f9fa',
    'fontFamily': 'Arial, sans-serif',
    'padding': '20px',
    'minHeight': '100vh',
    'display': 'flex',
    'flexDirection': 'column',
    'alignItems': 'center',
    'justifyContent': 'center'
}, children=[

    # Title Section
    html.Div(style={'textAlign': 'center'}, children=[
        html.H1("📰 News Category Predictor", style={
            'color': '#343a40',
            'fontSize': '36px',
            'marginBottom': '20px',
            'fontWeight': 'bold',
            'letterSpacing': '1px'
        }),
        html.P("Easily categorize your news headlines into predefined categories!", style={
            'fontSize': '18px',
            'color': '#6c757d',
            'marginBottom': '30px'
        }),
    ]),

    # Input Section
    html.Div(style={
        'textAlign': 'center',
        'width': '80%',
        'maxWidth': '600px',
        'marginBottom': '20px'
    }, children=[
        html.Label("Enter a news headline:", style={
            'fontSize': '20px',
            'color': '#495057',
            'marginBottom': '10px',
            'display': 'block'
        }),
        dcc.Input(id='headline-input', type='text', placeholder="Type your headline here...", style={
            'width': '100%',
            'padding': '12px',
            'fontSize': '16px',
            'border': '1px solid #ced4da',
            'borderRadius': '8px',
            'marginBottom': '20px',
            'boxShadow': '0 2px 4px rgba(0, 0, 0, 0.1)',
            'outline': 'none'
        }),
        html.Button('Predict', id='predict-button', n_clicks=0, style={
            'backgroundColor': '#007bff',
            'color': 'white',
            'padding': '12px 20px',
            'fontSize': '18px',
            'border': 'none',
            'borderRadius': '8px',
            'cursor': 'pointer',
            'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.1)',
            'transition': 'all 0.3s ease-in-out',
            'outline': 'none'
        }),
    ]),

    # Prediction Output Section
    html.Div(id='prediction-output', style={
        'fontSize': '22px',
        'color': '#495057',
        'fontWeight': 'bold',
        'marginTop': '30px',
        'padding': '15px',
        'border': '1px solid #dee2e6',
        'borderRadius': '8px',
        'backgroundColor': '#ffffff',
        'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.1)',
        'display': 'none'  # Hidden by default
    }),

    # Graph Section
    dcc.Graph(id='category-distribution', style={'marginTop': '40px', 'width': '80%', 'maxWidth': '800px'}),

    # Footer Section
    html.Div(style={
        'marginTop': '30px',
        'color': '#adb5bd',
        'fontSize': '16px',
        'textAlign': 'center'
    }, children=[
        html.P("Built by Arjun | 2025")
    ])
])

# Define callback to predict category and update graph
@app.callback(
    [Output('prediction-output', 'children'),
     Output('prediction-output', 'style'),
     Output('category-distribution', 'figure')],
    [Input('predict-button', 'n_clicks')],
    [Input('headline-input', 'value')]
)
def predict_category(n_clicks, headline):
    if n_clicks > 0 and headline:
        # Preprocess input headline
        processed_headline = preprocess_text(headline)
        
        # Vectorize the headline
        X_new = vectorizer.transform([processed_headline])
        
        # Predict category
        prediction = model.predict(X_new)
        
        # Map prediction to category name
        category_map_reverse = {0: 'Sports', 1: 'Tech', 2: 'Business', 3: 'Health', 
                                4: 'Entertainment', 5: 'Politics', 6: 'World News', 7: 'Lifestyle'}
        
        predicted_category = category_map_reverse.get(prediction[0], 'Unknown')
        
        # Update prediction count
        prediction_counts[predicted_category] += 1
        
        # Create the bar chart for category distribution
        categories = list(prediction_counts.keys())
        counts = list(prediction_counts.values())

        figure = {
            'data': [
                go.Bar(
                    x=categories,
                    y=counts,
                    marker={'color': '#007bff'}
                )
            ],
            'layout': go.Layout(
                title='Category Distribution of Predictions',
                xaxis={'title': 'Categories'},
                yaxis={'title': 'Count of Predictions'},
                template='plotly_white'
            )
        }
        
        return f"Predicted Category: {predicted_category}", {'display': 'block'}, figure
    else:
        # Hide the output if no input or button click
        return "", {'display': 'none'}, {
            'data': [],
            'layout': go.Layout(
                title='Category Distribution of Predictions',
                xaxis={'title': 'Categories'},
                yaxis={'title': 'Count of Predictions'},
                template='plotly_white'
            )
        }

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
