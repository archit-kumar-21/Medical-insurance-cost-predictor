import pandas as pd

def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Ensure expected columns exist (optional safety)
    expected = {'age','sex','bmi','children','smoker','region','charges'}
    missing = expected.difference(set(df.columns))
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")
    # Normalize text columns to lowercase/stripped
    for col in ['sex','smoker','region']:
        df[col] = df[col].astype(str).str.strip().str.lower()
    return df
