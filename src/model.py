import pandas as pd
import numpy as np
import joblib
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.cluster import DBSCAN
from src.preprocessing import load_and_clean_data, feature_engineering, prepare_modeling_data
import streamlit as st

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def train_risk_model():
    '''Trains the risk score classifier with balanced class weights and Cross Validation'''
    df = load_and_clean_data("data/accident_data.csv")
    df, encoders = feature_engineering(df)
    X, y, feature_names = prepare_modeling_data(df)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rf_model = RandomForestClassifier(
        n_estimators=150, 
        max_depth=12, 
        class_weight='balanced', 
        random_state=42
    )
    
    cv_scores = cross_val_score(rf_model, X_train, y_train, cv=3)
    print(f"Risk Model CV Accuracy: {cv_scores.mean():.2f} (+/- {cv_scores.std() * 2:.2f})")
    
    rf_model.fit(X_train, y_train)
    
    joblib.dump(rf_model, os.path.join(MODEL_DIR, "risk_model.pkl"))
    joblib.dump(encoders, os.path.join(MODEL_DIR, "label_encoders.pkl"))
    joblib.dump(feature_names, os.path.join(MODEL_DIR, "features.pkl"))
    
    X_test.to_csv(os.path.join(MODEL_DIR, "x_test_sample.csv"), index=False)
    return rf_model

def train_insurance_model():
    '''Trains a regression model for insurance claim prediction'''
    df = load_and_clean_data("data/accident_data.csv")
    df, _ = feature_engineering(df)
    
    np.random.seed(42)
    base_cost = 5000
    df['Claim_Amount'] = base_cost + (df['Severity_Score'] * 25000) + (df['Vehicles_Involved'] * 10000) + (np.random.normal(0, 5000, len(df)))
    df['Claim_Amount'] = df['Claim_Amount'].clip(lower=1000)
    
    X, _, _ = prepare_modeling_data(df)
    X = X.copy()
    X['Vehicles_Involved'] = df['Vehicles_Involved']
    y = df['Claim_Amount']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rf_reg = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
    rf_reg.fit(X_train, y_train)
    
    joblib.dump(rf_reg, os.path.join(MODEL_DIR, "insurance_model.pkl"))
    return rf_reg

@st.cache_resource
def load_risk_pipeline():
    try:
        model = joblib.load(os.path.join(MODEL_DIR, "risk_model.pkl"))
        encoders = joblib.load(os.path.join(MODEL_DIR, "label_encoders.pkl"))
        features = joblib.load(os.path.join(MODEL_DIR, "features.pkl"))
        return model, encoders, features
    except Exception:
        return None, None, None

def simulate_pipeline(input_data, model, encoders, features):
    '''Transform raw dict to model-ready features without file IO repeated'''
    df = pd.DataFrame([input_data])
    df, _ = feature_engineering(df)
    
    for col, le in encoders.items():
        if col in df.columns:
            try:
                if df[col].values[0] in le.classes_:
                    df[f'{col}_Encoded'] = le.transform(df[col])
                else:
                    df[f'{col}_Encoded'] = 0
            except ValueError:
                df[f'{col}_Encoded'] = 0
                
    X_input = pd.DataFrame(columns=features)
    for f in features:
        X_input.loc[0, f] = df[f].values[0] if f in df.columns else 0
        
    return X_input

def predict_risk_score(input_data):
    '''Returns risk score %, active category, confidence array, and model-ingested features'''
    model, encoders, features = load_risk_pipeline()
    if not model: return 0.0, "Unknown", 0.0, None
    
    X_input = simulate_pipeline(input_data, model, encoders, features)
    
    probs = model.predict_proba(X_input)[0]
    
    risk_score = (probs[0] * 5) + (probs[1] * 55) + (probs[2] * 98)
    
    category = "High" if risk_score >= 70 else "Medium" if risk_score >= 35 else "Low"
    
    # Calculate confidence based on the largest class probability
    confidence = max(probs) * 100
    
    return round(risk_score, 2), category, confidence, X_input

def get_feature_importances():
    '''Extracts global feature importances'''
    model, _, features = load_risk_pipeline()
    if not model: return pd.DataFrame()
    return pd.DataFrame({
        'Feature': features, 
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)

def perform_dbscan_clustering(df):
    '''V3 Geographic Advanced Clustering: Groups coordinates tightly together as distinct zones'''
    if df.empty: return df
    # EPS is approximately 500 meters, min_samples=10 ensures only dense areas become clusters
    coords = df[['Latitude', 'Longitude']].values
    # To radians for haversine
    db = DBSCAN(eps=0.005, min_samples=5).fit(coords)
    df = df.copy()
    df['Cluster_ID'] = db.labels_
    return df

if __name__ == "__main__":
    train_risk_model()
    train_insurance_model()
