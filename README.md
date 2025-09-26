# Medical Insurance Cost Predictor – End-to-End ML Web App

An end-to-end project to predict medical insurance charges using demographic and health features.
Includes: EDA, preprocessing with scikit-learn `ColumnTransformer`, model training with GridSearchCV,
evaluation plots, model persistence, and a Flask web app for local deployment.

## Project Structure
```
medical_insurance_cost_predictor/
├── app.py                     # Flask web app for local deployment
├── requirements.txt
├── README.md
├── data/
│   └── insurance.csv          # dataset (copied here)
├── src/
│   ├── train_model.py         # training + evaluation + artifact saving
│   └── utils.py               # small helpers
├── models/
│   ├── best_model.joblib
│   ├── preprocess_pipeline.joblib
│   └── full_pipeline.joblib
├── artifacts/
│   ├── metrics.json
│   ├── actual_vs_predicted.png
│   ├── residuals_hist.png
│   ├── charges_distribution.png
│   ├── charges_vs_age.png
│   ├── charges_vs_bmi.png
│   ├── charges_vs_smoker_box.png
│   ├── charges_boxplot.png
│   └── bmi_boxplot.png
├── templates/
│   └── index.html
└── static/
    └── style.css
```

## How to run locally

1. **Create a virtual environment and install deps**
   ```bash
   cd medical_insurance_cost_predictor
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # macOS/Linux:
   # source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Train the models and generate artifacts**
   ```bash
   python src/train_model.py
   ```
   This will create `models/` and `artifacts/` with all saved files.

3. **Start the web app (localhost)**
   ```bash
   python app.py
   ```
   Visit http://127.0.0.1:5000 in your browser.

4. **API usage (JSON)**
   ```bash
   curl -X POST http://127.0.0.1:5000/predict      -H "Content-Type: application/json"      -d '{
           "age": 29,
           "sex": "female",
           "bmi": 27.9,
           "children": 0,
           "smoker": "no",
           "region": "southwest"
         }'
   ```

## Notes
- Categorical features are one-hot encoded with `handle_unknown="ignore"`.
- Numerical features (`age`, `bmi`, `children`) are standardized.
- Models tried: Linear Regression (baseline), Decision Tree, Random Forest (with GridSearchCV).
- Best model is chosen on test RMSE.
- The Flask app loads the **full pipeline** so you can send raw inputs.
