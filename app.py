from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__)

PROJECT_ROOT = os.path.dirname(__file__)
MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'full_pipeline.joblib')

# Load the full pipeline (preprocess + model)
pipeline = joblib.load("models/full_pipeline.joblib")

def prepare_input_from_form(form):
    # Get and normalize inputs
    age = int(form.get('age', 0))
    sex = str(form.get('sex', '')).strip().lower()
    bmi = float(form.get('bmi', 0.0))
    children = int(form.get('children', 0))
    smoker = str(form.get('smoker', '')).strip().lower()
    region = str(form.get('region', '')).strip().lower()

    row = {
        'age': age,
        'sex': sex,
        'bmi': bmi,
        'children': children,
        'smoker': smoker,
        'region': region
    }
    return pd.DataFrame([row])

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', prediction=None)

@app.route('/predict-form', methods=['POST'])
def predict_form():
    try:
        X = prepare_input_from_form(request.form)
        pred = pipeline.predict(X)[0]
        return render_template('index.html', prediction=round(float(pred), 2))
    except Exception as e:
        return render_template('index.html', prediction=None, error=str(e))

@app.route('/predict', methods=['POST'])
def predict_api():
    try:
        data = request.get_json(force=True)
        if isinstance(data, dict):
            data = [data]
        df = pd.DataFrame(data)
        # normalize categorical inputs
        for col in ['sex', 'smoker', 'region']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.lower()
        pred = pipeline.predict(df)
        return jsonify({'predictions': [float(x) for x in pred]})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    import webbrowser
    webbrowser.open("http://127.0.0.1:5000/")
    app.run(host='127.0.0.1', port=5000, debug=True)
