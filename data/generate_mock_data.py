import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

np.random.seed(42)
random.seed(42)

NUM_RECORDS = 10500

CITIES = {
    "Mumbai": {"lat": 19.0760, "lon": 72.8777},
    "Delhi": {"lat": 28.7041, "lon": 77.1025},
    "Bangalore": {"lat": 12.9716, "lon": 77.5946},
    "Chennai": {"lat": 13.0827, "lon": 80.2707},
    "Hyderabad": {"lat": 17.3850, "lon": 78.4867}
}

WEATHER = ["Clear", "Rain", "Fog", "Overcast"]
ROAD_TYPE = ["Highway", "Urban", "Rural"]
TRAFFIC_DENSITY = ["Low", "Medium", "High"]
CAUSES = ["Speeding", "Drunk Driving", "Weather Conditions", "Pothole", "Signal Violation", "Vehicle Breakdown"]

def generate_accident_data():
    data = []
    start_date = datetime(2023, 1, 1)
    
    for _ in range(NUM_RECORDS):
        # 1. Date and Time (skewed towards evening/night)
        days_offset = random.randint(0, 364)
        date = (start_date + timedelta(days=days_offset)).date()
        
        # Base hour distribution: higher probability for 17:00 to 02:00
        hour_probs = [0.01]*6 + [0.02]*4 + [0.05]*7 + [0.08]*5 + [0.055]*2
        hour = np.random.choice(range(24), p=hour_probs)
        minute = random.randint(0, 59)
        time_str = f"{hour:02d}:{minute:02d}"
        
        # 2. City and Location
        city = random.choice(list(CITIES.keys()))
        lat = CITIES[city]["lat"] + random.uniform(-0.1, 0.1)
        lon = CITIES[city]["lon"] + random.uniform(-0.1, 0.1)
        
        # 3. Time-dependent Features
        if 6 <= hour < 18:
            lighting = "Daylight"
        elif 18 <= hour < 20 or 5 <= hour < 6:
            lighting = "Dawn/Dusk"
        else:
            lighting = np.random.choice(["Night (Lit)", "Night (Unlit)"], p=[0.7, 0.3])
            
        # 4. Traffic Density and Weather
        traffic = np.random.choice(TRAFFIC_DENSITY, p=[0.2, 0.4, 0.4]) # High/Medium more common
        weather = np.random.choice(WEATHER, p=[0.6, 0.2, 0.1, 0.1])
        
        # 5. Road Type
        road = np.random.choice(ROAD_TYPE, p=[0.3, 0.6, 0.1])
        
        # 6. Cause
        base_cause_probs = [0.4, 0.1, 0.1, 0.1, 0.2, 0.1]
        if weather in ["Rain", "Fog"]:
            base_cause_probs = [0.2, 0.1, 0.4, 0.1, 0.1, 0.1] # Weather becomes huge factor
        cause = np.random.choice(CAUSES, p=base_cause_probs)
        
        # 7. Severity (Correlated logic)
        # Higher on highways, higher at night unlit, higher with speeding/trucks
        sev_score = 0
        if road == "Highway": sev_score += 2
        if lighting == "Night (Unlit)": sev_score += 2
        if cause in ["Speeding", "Drunk Driving"]: sev_score += 1
        if weather == "Fog": sev_score += 1
        
        if sev_score >= 4:
            severity = "High"
        elif sev_score >= 2:
            severity = "Medium"
        else:
            severity = "Low"
            
        # 8. Vehicles Involved
        if severity == "High":
            vehicles = np.random.choice([2, 3, 4, 5], p=[0.4, 0.3, 0.2, 0.1])
        else:
            vehicles = np.random.choice([1, 2, 3], p=[0.3, 0.6, 0.1])
            
        data.append({
            "Date": date.strftime("%Y-%m-%d"),
            "Time": time_str,
            "City": city,
            "Latitude": round(lat, 5),
            "Longitude": round(lon, 5),
            "Severity": severity,
            "Weather": weather,
            "Road_Type": road,
            "Lighting": lighting,
            "Traffic_Density": traffic,
            "Vehicles_Involved": vehicles,
            "Cause": cause
        })
        
    df = pd.DataFrame(data)
    df.to_csv("data/accident_data.csv", index=False)
    print(f"Generated {len(df)} records for accident data.")

def generate_social_media_data():
    tweets_templates = [
        ("Terrible traffic near the highway intersection today. Avoid!", -0.6),
        ("Another accident due to poor lighting on the main road.", -0.8),
        ("Roads are extremely slippery due to rain. Drive safe.", -0.3),
        ("Huge pothole caused a bike crash just now.", -0.9),
        ("Traffic is moving smoothly this morning.", 0.5),
        ("Can the city please fix the signals? It's causing chaos.", -0.7),
        ("Visibility is zero because of the fog. Very dangerous.", -0.8),
        ("Drunk driver caught by police before causing harm. Good job.", 0.4),
        ("Severe accident on the bypass road. Ambulances rushing.", -1.0)
    ]
    
    data = []
    start_date = datetime(2023, 1, 1)
    
    for _ in range(5000):
        days_offset = random.randint(0, 364)
        ts = start_date + timedelta(days=days_offset, hours=random.randint(0, 23))
        
        city = random.choice(list(CITIES.keys()))
        template, base_sentiment = random.choice(tweets_templates)
        
        # Add a tiny bit of noise to sentiment to make it continuous
        sentiment = min(max(base_sentiment + random.uniform(-0.1, 0.1), -1.0), 1.0)
        
        data.append({
            "City": city,
            "Timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "Text": template,
            "Sentiment": round(sentiment, 2)
        })
        
    df = pd.DataFrame(data)
    df.to_csv("data/social_media_data.csv", index=False)
    print(f"Generated {len(df)} records for social media data.")

if __name__ == "__main__":
    print("Generating mock data...")
    generate_accident_data()
    generate_social_media_data()
    print("Mock data generation complete!")
