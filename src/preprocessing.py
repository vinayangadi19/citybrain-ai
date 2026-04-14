import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def load_and_clean_data(filepath):
    '''Loads csv data and performs basic cleaning'''
    df = pd.read_csv(filepath)
    df = df.dropna() # For robust generic datasets
    return df

def feature_engineering(df):
    '''Extracts essential features from raw columns'''
    df = df.copy()
    
    # 1. Date features
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Is_Weekend'] = df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
    
    # 2. Time features
    if 'Time' in df.columns:
        df['Hour'] = df['Time'].apply(lambda x: int(str(x).split(':')[0]) if ':' in str(x) else 0)
    
    # 3. Categorical encoding
    categorical_cols = ['City', 'Weather', 'Road_Type', 'Lighting', 'Traffic_Density', 'Cause']
    label_encoders = {}
    
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[f'{col}_Encoded'] = le.fit_transform(df[col])
            label_encoders[col] = le
            
    # 4. Target variable encoding
    severity_map = {"Low": 0, "Medium": 1, "High": 2}
    if 'Severity' in df.columns and df['Severity'].dtype == object:
        df['Severity_Score'] = df['Severity'].map(severity_map)
        
    return df, label_encoders

def prepare_modeling_data(df):
    '''Prepares the dataframe specifically for model training/prediction'''
    # Assume we use Weather, Road_Type, Lighting, Traffic, and Hour for Risk Prediction
    features = ['Hour', 'Is_Weekend']
    
    # Add encoded features if available
    for col in ['Weather_Encoded', 'Road_Type_Encoded', 'Lighting_Encoded', 'Traffic_Density_Encoded', 'City_Encoded']:
        if col in df.columns:
            features.append(col)
            
    X = df[features]
    
    # Risk score can be modeled on Severity_Score or a composite target
    # Here we simulate a risk probability based on severity and involved vehicles
    y = df['Severity_Score']
    return X, y, features
