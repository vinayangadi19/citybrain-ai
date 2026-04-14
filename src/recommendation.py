import pandas as pd

def generate_infrastructure_recommendations(row):
    '''
    Analyzes infrastructure-factors to yield UI-ready recommendations.
    Outputs: List of tuples -> (Priority, Title, Description, Icon)
    '''
    recs = []
    
    road_type = row.get('Road_Type', 'Unknown')
    lighting = row.get('Lighting', 'Unknown')
    traffic = row.get('Traffic_Density', 'Unknown')
    cause = row.get('Cause', 'Unknown')
    weather = row.get('Weather', 'Unknown')
    
    # 1. Lighting Recommendations
    if lighting == 'Night (Unlit)':
        recs.append(("HIGH", "Street Illumination", "Immediate street lighting installation recommended. Poor visibility is functioning as a severe risk multiplier.", "💡"))
        
    # 2. Road Condition / Pothole
    if cause == 'Pothole':
        recs.append(("MEDIUM", "Structural Maintenance", "Frequent pothole-related accidents. Road resurfacing required and structural audits suggested.", "🚧"))
        
    # 3. Traffic and Signals
    if traffic == 'High' and cause == 'Signal Violation':
        recs.append(("HIGH", "Signal Enhancement", "Install smart traffic lights or automated speed cameras. Intersection throughput exceeding safe control.", "🚦"))
    elif traffic == 'High' and road_type == 'Urban':
        recs.append(("LOW", "Timing Optimization", "Optimization of signal timing recommended to alleviate Urban density friction.", "⏱️"))
        
    # 4. Highway Speed
    if road_type == 'Highway' and cause in ['Speeding', 'Drunk Driving']:
        recs.append(("HIGH", "Enforcement Protocol", "Implement radar speed boards or increase patrol presence. Excessive speed bounds breached.", "🚓"))
        
    # 5. Weather Mitigation
    if weather in ['Fog', 'Rain'] and road_type == 'Highway':
        recs.append(("LOW", "VMS Alerting", "Add variable message signs (VMS) warning drivers of adverse visibility.", "🌦️"))
        
    if not recs:
        recs.append(("LOW", "Nominal Status", "No immediate critical infrastructure failures detected. Continue routine periodic monitoring.", "✅"))
        
    return recs

def aggregated_recommendations(df, city):
    '''
    Generates high-level city-wide insights
    Outputs: List of tuples -> (Priority, Title, Description, Icon)
    '''
    city_df = df[df['City'] == city]
    if city_df.empty: return []
    
    top_cause = city_df['Cause'].value_counts().index[0]
    unlit_accidents = len(city_df[city_df['Lighting'] == 'Night (Unlit)'])
    
    insights = []
    insights.append(("MEDIUM", "Macro Friction Source", f"City-wide primary friction is driven largely by {top_cause}.", "📊"))
    
    if unlit_accidents > (len(city_df) * 0.1):
        insights.append(("HIGH", "Capital Upgrade Priority", f"Major liability: {unlit_accidents} incidents mapped to unlit roads. City-wide lighting drive required.", "💡"))
        
    return insights
