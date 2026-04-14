import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

def explain_prediction(input_data_df):
    '''
    Uses SHAP TreeExplainer to explain the prediction from the Random Forest model.
    Also translates the top driving features into plain human-readable English.
    '''
    try:
        model = joblib.load("models/risk_model.pkl")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_data_df)
        
        # Determine the array corresponding to the High Risk class (class 2)
        if isinstance(shap_values, list):
            vals = shap_values[2] 
            shap_vals_row = vals[0] if len(vals.shape) > 1 else vals
        else:
            # Handle ndarray of shape (samples, features, classes)
            if len(shap_values.shape) == 3:
                vals = shap_values[:, :, 2]
                shap_vals_row = vals[0]
            else:
                vals = shap_values
                shap_vals_row = vals[0] if len(vals.shape) > 1 else vals
                
        # 1. SHAP Human Readable Text Generation
        # Get the feature names and values for this specific instance
        feature_names = input_data_df.columns.tolist()
        shap_vals_row = np.array(shap_vals_row).flatten()
        
        # Combine into dataframe and sort
        contributions = pd.DataFrame({
            'Feature': feature_names,
            'SHAP_Value': shap_vals_row
        })
        
        # Sort by absolute impact
        contributions['Abs_Impact'] = contributions['SHAP_Value'].abs()
        top_factors = contributions.sort_values(by='Abs_Impact', ascending=False).head(3)
        
        explanations = []
        for _, row in top_factors.iterrows():
            if row['SHAP_Value'] > 0.01:
                explanations.append(f"🔴 **{row['Feature']}** strongly increased the risk probability (Impact factor: +{row['SHAP_Value']:.3f}).")
            elif row['SHAP_Value'] < -0.01:
                explanations.append(f"🟢 **{row['Feature']}** helped reduce the risk substantially (Impact factor: {row['SHAP_Value']:.3f}).")
                
        if not explanations:
            explanations.append("⚪ Risk is balanced; no single factor is disproportionately dominating the prediction.")
            
        # 2. Visual Chart Generation
        fig, ax = plt.subplots(figsize=(8, 3))
        shap.summary_plot(vals, input_data_df, plot_type="bar", show=False, color="#dc3545")
        plt.tight_layout()
        
        return fig, explanations
        
    except Exception as e:
        st.error(f"Error generating Explainable AI diagnostics: {str(e)}")
        return None, []
