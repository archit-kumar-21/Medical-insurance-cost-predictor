import os, json
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt

from utils import load_data

#defining the foleder structure
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'insurance.csv')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, 'artifacts')

#this will autocreate folder if not present
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

def plot_and_save_charts(df: pd.DataFrame):
    # 1) Charges distribution
    plt.figure()
    df['charges'].hist(bins=30)
    plt.title('Charges Distribution')
    plt.xlabel('charges')
    plt.ylabel('frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACTS_DIR, 'charges_distribution.png'))
    plt.close()

    # 2) Charges vs Age (scatter)
    plt.figure()
    plt.scatter(df['age'], df['charges'], alpha=0.5)
    plt.title('Charges vs Age')
    plt.xlabel('age')
    plt.ylabel('charges')
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACTS_DIR, 'charges_vs_age.png'))
    plt.close()

    # 3) Charges vs BMI (scatter)
    plt.figure()
    plt.scatter(df['bmi'], df['charges'], alpha=0.5)
    plt.title('Charges vs BMI')
    plt.xlabel('bmi')
    plt.ylabel('charges')
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACTS_DIR, 'charges_vs_bmi.png'))
    plt.close()

    # 4) Charges vs Smoker (boxplot): convert smoker yes/no -> 1/0
    plt.figure()
    smoker_map = df['smoker'].map({'yes':1,'no':0})
    data = [df.loc[smoker_map==0, 'charges'].values,
            df.loc[smoker_map==1, 'charges'].values]
    plt.boxplot(data, labels=['no','yes'])
    plt.title('Charges by Smoker (boxplot)')
    plt.xlabel('smoker')
    plt.ylabel('charges')
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACTS_DIR, 'charges_vs_smoker_box.png'))
    plt.close()

    # 5) Boxplot for Charges (outlier detection)
    plt.figure()
    plt.boxplot(df['charges'].values, vert=True)
    plt.title('Charges Boxplot')
    plt.ylabel('charges')
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACTS_DIR, 'charges_boxplot.png'))
    plt.close()

    # 6) Boxplot for BMI (outlier detection)
    plt.figure()
    plt.boxplot(df['bmi'].values, vert=True)
    plt.title('BMI Boxplot')
    plt.ylabel('bmi')
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACTS_DIR, 'bmi_boxplot.png'))
    plt.close()

def build_preprocessor():
    numeric_features = ['age', 'bmi', 'children']
    categorical_features = ['sex', 'smoker', 'region']

    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    return preprocessor

def evaluate_and_save_plots(y_true, y_pred):
    # Actual vs Predicted
    plt.figure()
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.title('Actual vs Predicted Charges')
    plt.xlabel('actual')
    plt.ylabel('predicted')
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACTS_DIR, 'actual_vs_predicted.png'))
    plt.close()

    # Residuals histogram
    residuals = y_true - y_pred
    plt.figure()
    plt.hist(residuals, bins=30)
    plt.title('Residuals Distribution')
    plt.xlabel('residual')
    plt.ylabel('frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACTS_DIR, 'residuals_hist.png'))
    plt.close()

def main():
    df = load_data(DATA_PATH)
    plot_and_save_charts(df)

    X = df.drop(columns=['charges'])
    y = df['charges']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Preprocessing pipeline
    preprocessor = build_preprocessor()

    # Baseline: Linear Regression
    lr_pipeline = Pipeline(steps=[
        ('preprocess', preprocessor),
        ('model', LinearRegression())
    ])
    lr_pipeline.fit(X_train, y_train)
    lr_pred = lr_pipeline.predict(X_test)
    lr_mse = mean_squared_error(y_test, lr_pred)
    lr_rmse = np.sqrt(lr_mse)
    lr_r2 = r2_score(y_test, lr_pred)

    # Decision Tree with GridSearchCV
    dt_pipeline = Pipeline(steps=[
        ('preprocess', preprocessor),
        ('model', DecisionTreeRegressor(random_state=42))
    ])
    dt_param_grid = {
        'model__max_depth': [3, 5, 7, 9, None],
        'model__min_samples_split': [2, 5, 10],
        'model__min_samples_leaf': [1, 2, 5]
    }
    dt_gs = GridSearchCV(dt_pipeline, dt_param_grid, cv=5, scoring='neg_root_mean_squared_error', n_jobs=-1)
    dt_gs.fit(X_train, y_train)
    dt_pred = dt_gs.predict(X_test)
    dt_rmse = np.sqrt(mean_squared_error(y_test, dt_pred))
    dt_r2 = r2_score(y_test, dt_pred)

    
    rf_gs.fit(X_train, y_train)
    rf_pred = rf_gs.predict(X_test)
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
    rf_r2 = r2_score(y_test, rf_pred)

    # Choose best by RMSE
    results = {
        'linear_regression': {'rmse': float(lr_rmse), 'r2': float(lr_r2)},
        'decision_tree': {'rmse': float(dt_rmse), 'r2': float(dt_r2), 'best_params': dt_gs.best_params_},
        'random_forest': {'rmse': float(rf_rmse), 'r2': float(rf_r2), 'best_params': rf_gs.best_params_}
    }

    # Determine best
    best_name = min(results, key=lambda k: results[k]['rmse'])
    if best_name == 'linear_regression':
        best_pipeline = lr_pipeline
        best_estimator = lr_pipeline.named_steps['model']
    elif best_name == 'decision_tree':
        best_pipeline = dt_gs.best_estimator_
        best_estimator = dt_gs.best_estimator_.named_steps['model']
    else:
        best_pipeline = rf_gs.best_estimator_
        best_estimator = rf_gs.best_estimator_.named_steps['model']

    # Evaluate best and save plots
    best_pred = best_pipeline.predict(X_test)
    evaluate_and_save_plots(y_test, best_pred)

    # Save metrics
    with open(os.path.join(ARTIFACTS_DIR, 'metrics.json'), 'w') as f:
        json.dump({'results': results, 'best_model': best_name}, f, indent=2)

    # Save preprocessing pipeline separately
    preprocessor_fitted = best_pipeline.named_steps['preprocess']
    joblib.dump(preprocessor_fitted, o# Random Forest with GridSearchCV
    rf_pipeline = Pipeline(steps=[
        ('preprocess', preprocessor),
        ('model', RandomForestRegressor(random_state=42))
    ])
    rf_param_grid = {
        'model__n_estimators': [100, 300],
        'model__max_depth': [None, 5, 10, 15],
        'model__min_samples_leaf': [1, 2, 5]
    }
    rf_gs = GridSearchCV(rf_pipeline, rf_param_grid, cv=5, scoring='neg_root_mean_squared_error', n_jobs=-1)s.path.join(MODELS_DIR, 'preprocess_pipeline.joblib'))

    # Save model separately
    joblib.dump(best_estimator, os.path.join(MODELS_DIR, 'best_model.joblib'))

    # Save full pipeline (recommended for serving)
    joblib.dump(best_pipeline, os.path.join(MODELS_DIR, 'full_pipeline.joblib'))

    print('Training complete. Best model:', best_name)
    print('Metrics:', json.dumps(results, indent=2))

if __name__ == '__main__':
    main()
