# AI Road Safety Recommendation and Risk Intelligence System 🚦

A full-stack, AI-powered smart city platform designed to analyze, predict, and mitigate road accidents.

This system moves beyond basic analytics to provide:
1. **Real-time Risk Prediction:** Uses Machine Learning (Random Forest / XGBoost) to classify accident risks based on time, weather, and traffic data.
2. **Infrastructure Intelligence:** A rule-based recommender that dynamically identifies missing streetlights, dangerous highway stretches, and traffic hotspots.
3. **Spatial Intelligence:** Implements heatmaps to isolate crash hotspots.
4. **Social Sentiment Context:** Natural Language Processing (NLP) against social media complaints to find friction between public sentiment and actual crash data.
5. **Insurance Analytics:** Models estimated insurance claim amounts based on accident severity and vehicles involved.
6. **Explainable AI (XAI):** Uses SHAP values to explain the "why" behind every risk prediction directly in the UI.

## Project Structure
```text
road-safety-ai/
│
├── data/                    # Generated mock datasets (accident & social media)
├── models/                  # Saved Machine Learning models & encoders (.pkl)
├── src/                     # Core Processing Logic
│   ├── preprocessing.py     # Data pipeline & feature engineering
│   ├── eda.py               # Exploratory Data Analysis & Plotly visual generators
│   ├── model.py             # Random Forest training & inference logic
│   ├── sentiment.py         # NLP text sentiment analysis rules using TextBlob
│   ├── recommendation.py    # Rule-engine for Smart City Infrastructure insights
│   └── xai.py               # Explains ML decisions using SHAP
│
├── app.py                   # Main Streamlit Application Dashboard
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

## Setup Instructions

1. **Install Dependencies:**
   Ensure you have Python 3 installed. Run the following to install all necessary packages:
   ```bash
   pip install -r requirements.txt
   ```

2. **Generate Mock Data:**
   Run the build script to synthesize 10,000+ rows of realistic correlated accident data and social media sentiment.
   ```bash
   python data/generate_mock_data.py
   ```

3. **Train the Models:**
   Compile the data into trained Random Forest instances.
   ```bash
   python src/model.py
   ```

4. **Launch the Dashboard:**
   Run the beautiful Streamlit web interface.
   ```bash
   streamlit run app.py
   ```

## How to Use Real Data

By default, the `data/generate_mock_data.py` script ensures that you have completely functional data right away.
To use your own real-world data:
1. Replace `data/accident_data.csv` with your dataset.
2. Ensure columns roughly map to: `Date, Time, City, Latitude, Longitude, Severity, Weather, Road_Type, Lighting, Traffic_Density, Vehicles_Involved, Cause`.
3. Retrain the model.
4. View the dashboard!
