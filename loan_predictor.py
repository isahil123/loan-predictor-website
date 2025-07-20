import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize NLP tools
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Function for text preprocessing
def preprocess_text(text):
    if isinstance(text, str):
        tokens = word_tokenize(text.lower())
        tokens = [lemmatizer.lemmatize(ps.stem(token)) for token in tokens if token not in stop_words and token.isalpha()]
        return ' '.join(tokens)
    return ''

# Load and preprocess data
# def load_and_preprocess_data(file_path='loan_data.csv'):
#     try:
#         df = pd.read_csv(file_path)
#     except FileNotFoundError:
#         print("CSV file not found. Using provided data.")
#         # Load the provided CSV data (assuming it's saved as 'loan_data.csv')
#         # For this example, we assume the data is already in df from the provided snippet
#         df = pd.DataFrame({
#             'ApplicantIncome': [4146, 2028, None, 3955, ...],  # Replace with full data
#             'LoanAmount': [174, 184, None, 475, ...],
#             'CreditScore': [666, 700, 740, 466, ...],
#             'Education': ['Not Graduate', 'Graduate', 'Graduate', 'Graduate', ...],
#             'SelfEmployed': ['Yes', 'No', 'No', 'Yes', ...],
#             'LoanApproved': [1, None, 1, 1, ...]
#         })  # Note: Replace with actual data loading

#     # Clean LoanApproved: Convert invalid values to NaN, keep only 0 or 1
#     df['LoanApproved'] = pd.to_numeric(df['LoanApproved'], errors='coerce')
#     df['LoanApproved'] = df['LoanApproved'].apply(lambda x: x if x in [0, 1] else None)

#     # Drop rows where LoanApproved is NaN
#     df = df.dropna(subset=['LoanApproved'])

#     num_cols = ['ApplicantIncome', 'LoanAmount', 'CreditScore']
#     cat_cols = ['Education', 'SelfEmployed']

#     # Impute missing numerical values
#     num_imputer = SimpleImputer(strategy='mean')
#     df[num_cols] = num_imputer.fit_transform(df[num_cols])

#     # Impute missing categorical values
#     cat_imputer = SimpleImputer(strategy='most_frequent')
#     df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

#     # Process text data
#     df['education_processed'] = df['Education'].apply(preprocess_text)
#     df['SelfEmployed'] = df['SelfEmployed'].map({'Yes': 1, 'No': 0})

#     # TF-IDF for text features
#     tfidf = TfidfVectorizer(max_features=100)
#     text_features = tfidf.fit_transform(df['education_processed']).toarray()

#     # Combine numeric and text features
#     numeric_features = df[['ApplicantIncome', 'LoanAmount', 'CreditScore', 'SelfEmployed']].values
#     X = pd.DataFrame(numeric_features, columns=['ApplicantIncome', 'LoanAmount', 'CreditScore', 'SelfEmployed'])
#     X = pd.concat([X, pd.DataFrame(text_features, columns=[f'text_{i}' for i in range(text_features.shape[1])])], axis=1)

#     # Scale features
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)

#     return X_scaled, df['LoanApproved'], scaler, tfidf, X.columns

# Train models
# Load and preprocess data
def load_and_preprocess_data(file_path='loan_data.csv'):
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print("CSV file not found. Using sample data.")
        # Sample data as fallback (replace with your CSV data if needed)
        data = {
            'ApplicantIncome': [50000, 60000, 45000, 80000, 30000],
            'LoanAmount': [200000, 150000, 100000, 300000, 80000],
            'CreditScore': [700, 650, 720, 680, 600],
            'Education': ['Bachelor', 'Master', 'High School', 'PhD', 'Bachelor'],
            'SelfEmployed': ['No', 'Yes', 'No', 'Yes', 'No'],
            'LoanApproved': [1, 0, 1, 1, 0]
        }
        df = pd.DataFrame(data)

    print("Initial dataset shape:", df.shape)
    print("Initial LoanApproved values:", df['LoanApproved'].value_counts(dropna=False))

    # Clean LoanApproved: Convert invalid values to NaN, keep only 0 or 1
    df['LoanApproved'] = pd.to_numeric(df['LoanApproved'], errors='coerce')
    df = df[df['LoanApproved'].isin([0, 1])].copy()  # Keep only valid 0/1 values
    df = df.dropna(subset=['LoanApproved'])  # Ensure no NaN in LoanApproved

    print("After cleaning LoanApproved, dataset shape:", df.shape)
    print("Cleaned LoanApproved values:", df['LoanApproved'].value_counts(dropna=False))

    num_cols = ['ApplicantIncome', 'LoanAmount', 'CreditScore']
    cat_cols = ['Education', 'SelfEmployed']

    # Impute missing numerical values
    num_imputer = SimpleImputer(strategy='mean')
    df[num_cols] = num_imputer.fit_transform(df[num_cols])

    # Impute missing categorical values
    cat_imputer = SimpleImputer(strategy='most_frequent')
    df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

    # Process text data
    df['education_processed'] = df['Education'].apply(preprocess_text)
    df['SelfEmployed'] = df['SelfEmployed'].map({'Yes': 1, 'No': 0})

    # TF-IDF for text features
    tfidf = TfidfVectorizer(max_features=100)
    text_features = tfidf.fit_transform(df['education_processed']).toarray()

    # Combine numeric and text features
    numeric_features = df[['ApplicantIncome', 'LoanAmount', 'CreditScore', 'SelfEmployed']].values
    X = pd.DataFrame(numeric_features, columns=['ApplicantIncome', 'LoanAmount', 'CreditScore', 'SelfEmployed'])
    X = pd.concat([X, pd.DataFrame(text_features, columns=[f'text_{i}' for i in range(text_features.shape[1])])], axis=1)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Ensure y is aligned with X
    y = df['LoanApproved'].values

    print("Final X shape:", X_scaled.shape)
    print("Final y shape:", y.shape)
    print("Any NaN in y:", pd.isna(y).any())

    return X_scaled, y, scaler, tfidf, X.columns
def train_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    lr_model = LogisticRegression()
    lr_model.fit(X_train, y_train)
    lr_accuracy = accuracy_score(y_test, lr_model.predict(X_test))

    dt_model = DecisionTreeClassifier()
    dt_model.fit(X_train, y_train)
    dt_accuracy = accuracy_score(y_test, dt_model.predict(X_test))

    return lr_model, dt_model, lr_accuracy, dt_accuracy

# Prediction function
def predict_loan(model, scaler, tfidf, feature_names, income, loan_amount, credit_score, education, self_employed):
    education_processed = preprocess_text(education)
    text_features = tfidf.transform([education_processed]).toarray()
    self_employed_num = 1 if self_employed.lower() == 'yes' else 0

    numeric_features = [[income, loan_amount, credit_score, self_employed_num]]
    features = pd.DataFrame(numeric_features, columns=['ApplicantIncome', 'LoanAmount', 'CreditScore', 'SelfEmployed'])
    features = pd.concat([features, pd.DataFrame(text_features, columns=[f'text_{i}' for i in range(text_features.shape[1])])], axis=1)

    # Ensure feature order matches training
    features = features[feature_names]
    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0][1]

    return prediction, probability

# Initialize Dash app
app = dash.Dash(__name__)

# Load and train models
X, y, scaler, tfidf, feature_names = load_and_preprocess_data()
lr_model, dt_model, lr_accuracy, dt_accuracy = train_models(X, y)

# Dash layout
app.layout = html.Div([
    html.H1('Loan Approval Predictor', style={'textAlign': 'center'}),
    html.Div([
        html.Label('Applicant Income ($):'),
        dcc.Input(id='income', type='number', value=50000, min=0),
        
        html.Label('Loan Amount ($):'),
        dcc.Input(id='loan-amount', type='number', value=100000, min=0),
        
        html.Label('Credit Score:'),
        dcc.Input(id='credit-score', type='number', value=700, min=300, max=850),
        
        html.Label('Education Level:'),
        dcc.Input(id='education', type='text', value='Bachelor'),
        
        html.Label('Self Employed:'),
        dcc.Dropdown(id='self-employed',
                    options=[{'label': 'Yes', 'value': 'Yes'}, {'label': 'No', 'value': 'No'}],
                    value='No'),
        
        html.Label('Model Selection:'),
        dcc.Dropdown(id='model-selection',
                    options=[
                        {'label': f'Logistic Regression (Accuracy: {lr_accuracy:.2f})', 'value': 'lr'},
                        {'label': f'Decision Tree (Accuracy: {dt_accuracy:.2f})', 'value': 'dt'}
                    ],
                    value='lr'),
        
        html.Button('Predict', id='predict-button', n_clicks=0),
    ], style={'width': '50%', 'margin': 'auto'}),
    
    html.Div(id='prediction-output', style={'textAlign': 'center', 'marginTop': '20px'}),
    
    dcc.Graph(id='probability-graph'),
    
    dcc.Graph(id='model-accuracy-graph')
])

# Callback for prediction and probability graph
@app.callback(
    [Output('prediction-output', 'children'),
     Output('probability-graph', 'figure'),
     Output('model-accuracy-graph', 'figure')],
    [Input('predict-button', 'n_clicks')],
    [Input('income', 'value'),
     Input('loan-amount', 'value'),
     Input('credit-score', 'value'),
     Input('education', 'value'),
     Input('self-employed', 'value'),
     Input('model-selection', 'value')]
)
def update_prediction(n_clicks, income, loan_amount, credit_score, education, self_employed, model_selection):
    if n_clicks == 0:
        return '', {}, {}

    # Input validation
    try:
        income = float(income) if income is not None else 0
        loan_amount = float(loan_amount) if loan_amount is not None else 0
        credit_score = float(credit_score) if credit_score is not None else 300
        education = str(education) if education is not None else 'Bachelor'
        self_employed = str(self_employed) if self_employed is not None else 'No'
    except (ValueError, TypeError):
        return 'Error: Invalid input values', {}, {}

    model = lr_model if model_selection == 'lr' else dt_model

    try:
        prediction, probability = predict_loan(model, scaler, tfidf, feature_names,
                                              income, loan_amount, credit_score,
                                              education, self_employed)
        
        result = 'Approved' if prediction == 1 else 'Not Approved'
        output_text = f'Loan Status: {result} (Probability of Approval: {probability:.2%})'
        
        # Probability graph
        prob_fig = go.Figure(data=[
            go.Bar(x=['Approval Probability'], y=[probability], marker_color='blue')
        ])
        prob_fig.update_layout(
            yaxis_range=[0, 1],
            yaxis_title='Probability',
            title='Loan Approval Probability'
        )
        
        # Model accuracy chart
        acc_fig = {
            "type": "bar",
            "data": {
                "labels": ["Logistic Regression", "Decision Tree"],
                "datasets": [{
                    "label": "Model Accuracy",
                    "data": [lr_accuracy, dt_accuracy],
                    "backgroundColor": ["#36A2EB", "#FF6384"],
                    "borderColor": ["#36A2EB", "#FF6384"],
                    "borderWidth": 1
                }]
            },
            "options": {
                "scales": {
                    "y": {
                        "beginAtZero": True,
                        "max": 1,
                        "title": {
                            "display": True,
                            "text": "Accuracy"
                        }
                    },
                    "x": {
                        "title": {
                            "display": True,
                            "text": "Model"
                        }
                    }
                },
                "plugins": {
                    "title": {
                        "display": True,
                        "text": "Model Accuracy Comparison"
                    }
                }
            }
        }

        return output_text, prob_fig, acc_fig
    except Exception as e:
        return f'Error: {str(e)}', {}, {}

if __name__ == '__main__':
    app.run(debug=True)
    