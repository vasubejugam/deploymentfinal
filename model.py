from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import io
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder


app = Flask(__name__)

# Initialize a simple model
model = LinearRegression()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' in request.files:
            file = request.files['file']
            if file.filename != '':
                df = pd.read_csv(file)
                result = process_data(df)
                return render_template('result.html', result=result)
        elif 'data' in request.form:
            data = request.form['data']
            df = pd.read_csv(io.StringIO(data))
            result = process_data(df)
            return render_template('result.html', result=result)
    
    return render_template('index.html')

def process_data(df):
    # Check if dataframe is suitable
    if df.shape[1] < 2:
        return "The dataset must have at least two columns for prediction."

    # Separate features and target
    X = df.iloc[:, :-1].copy()  # Features
    y = df.iloc[:, -1].copy()   # Target

    # Encode categorical features
    X = encode_categorical(X)
    y = encode_categorical(y, target=True)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # Predict on the test set
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    
    return f"Mean Squared Error of the model: {mse}"

def encode_categorical(data, target=False):
    """
    Encode categorical data using Label Encoding.
    Handles both DataFrame and Series.
    """
    if isinstance(data, pd.DataFrame):
        for col in data.columns:
            if data[col].dtype == 'object':
                le = LabelEncoder()
                data[col] = le.fit_transform(data[col])
    elif isinstance(data, pd.Series):
        if data.dtype == 'object':
            le = LabelEncoder()
            data = le.fit_transform(data)
    return data

if __name__ == '__main__':
    app.run(debug=True)