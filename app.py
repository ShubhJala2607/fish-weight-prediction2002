from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd

# Load the model and columns
with open('model.pkl', 'rb') as f:
    model_data = pickle.load(f)
model = model_data["model"]
model_columns = model_data["columns"]

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_data = {
        "Length1": float(data['length1']),
        "Length2": float(data['length2']),
        "Length3": float(data['length3']),
        "Height": float(data['height']),
        "Width": float(data['width'])
    }
    
    # Convert input data to a DataFrame and align it with model columns
    input_df = pd.DataFrame([input_data])
    input_df = pd.get_dummies(input_df)
    input_df = input_df.reindex(columns=model_columns, fill_value=0)

    prediction = model.predict(input_df)[0]
    return jsonify({'weight': prediction})

if __name__ == '__main__':
    app.run(debug=True)

