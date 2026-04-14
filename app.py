import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import streamlit.components.v1 as components
from folium.plugins import HeatMap, MarkerCluster
import time

from src.eda import get_time_trend, get_causes_pie, get_hourly_trend, get_severity_by_weather, get_gauge_chart, get_top_blackspots, get_city_comparison, get_time_series_forecast, generate_insights
from src.model import predict_risk_score, get_feature_importances, perform_dbscan_clustering
from src.recommendation import aggregated_recommendations, generate_infrastructure_recommendations
from src.sentiment import get_city_sentiment_summary, process_social_media
from src.xai import explain_prediction
from src.preprocessing import feature_engineering
from src.ui_components import inject_premium_css, render_header, render_kpi, render_blackspot_card, render_recommendation_card

# --- PAGE CONFIG & CSS ---
st.set_page_config(page_title="CityBrain AI Platform", page_icon="🌐", layout="wide")
inject_premium_css()

# --- DATA CACHING ---
@st.cache_data
def load_data():
    try:
        acc_df = pd.read_csv("data/accident_data.csv")
        soc_df = process_social_media("data/social_media_data.csv")
        acc_df, _ = feature_engineering(acc_df)
        return acc_df, soc_df
    except Exception:
        return None, None

acc_df, soc_df = load_data()
if acc_df is None:
    st.error("Data files not found. Please run the generation script.")
    st.stop()

# --- SIDEBAR & GLOBAL CONTROLS ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c2/GitHub_Invertocat_Logo.svg/120px-GitHub_Invertocat_Logo.svg.png", width=60)
    st.markdown("## Global Controls")
    
    if st.button("🔄 Reset Filters"):
        st.session_state['city'] = "All"
        st.session_state['weather'] = "All"
        
    cities = ["All"] + list(acc_df['City'].unique())
    selected_city = st.selectbox("🏙️ Target City", cities, key='city')
    
    weathers = ["All"] + list(acc_df['Weather'].unique())
    selected_weather = st.selectbox("⛅ Weather Condition", weathers, key='weather')
    
    st.markdown("---")
    st.markdown("### Export & Reporting")
    
    @st.cache_data
    def convert_df(df): return df.to_csv(index=False).encode('utf-8')
    csv = convert_df(acc_df[(acc_df['City'] == selected_city) | (selected_city == "All")])
    st.download_button("📥 Download Filtered CSV", data=csv, file_name="citybrain_export.csv", mime="text/csv")
    st.download_button("📄 Download Summary Insight", data=f"Smart City Report for {selected_city}.", file_name="summary.txt")

render_header()

# Filter logic
filtered_df = acc_df.copy()
filtered_soc_df = soc_df.copy()
if selected_city != "All":
    filtered_df = filtered_df[filtered_df['City'] == selected_city]
    filtered_soc_df = filtered_soc_df[filtered_soc_df['City'] == selected_city]
if selected_weather != "All":
    filtered_df = filtered_df[filtered_df['Weather'] == selected_weather]

# --- SMART ALERT SYSTEM ---
if len(filtered_df) > 0:
    recent_highs = len(filtered_df[filtered_df['Severity'] == 'High'])
    if recent_highs > (len(filtered_df) * 0.35):
        st.markdown(f"<div style='background: rgba(255,123,114,0.1); border-left: 4px solid #ff7b72; padding: 15px; margin-bottom: 20px; border-radius: 4px; color: #fff;'>⚠️ <strong>ALERT:</strong> Unusually high ratio of Severe incidents detected within current parameters ({selected_city}). Immediate investigation recommended.</div>", unsafe_allow_html=True)

# --- NAVIGATION TABS ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Executive Dashboard", 
    "🗺️ Spatial Intelligence", 
    "🎛️ AI Simulator", 
    "🏗️ Infrastructure",
    "🧠 Explainable AI"
])

# --- TAB 1: EXECUTIVE DASHBOARD ---
with tab1:
    # 1. KPI Ribbon
    col1, col2, col3, col4 = st.columns(4)
    with col1: render_kpi("Total Incidents", f"{len(filtered_df):,}", "🚗", "#58a6ff")
    with col2: render_kpi("High Severity", f"{len(filtered_df[filtered_df['Severity'] == 'High']):,}", "⚠️", "#ff7b72")
    with col3: render_kpi("Top Friction Cause", filtered_df["Cause"].mode()[0] if not filtered_df.empty else "N/A", "🚦", "#d2a8ff")
    with col4: 
        avg_sent = filtered_soc_df['NLP_Sentiment'].mean() if not filtered_soc_df.empty else 0
        render_kpi("City Sentiment", f"{avg_sent:.2f}", "💬", "#3fb950" if avg_sent > 0 else "#ff7b72")

    # 2. Forecasting
    st.markdown("<h3 style='margin-top:20px; border-bottom:1px solid #30363d; padding-bottom:10px;'>30-Day Auto-Regressive Forecast</h3>", unsafe_allow_html=True)
    with st.container():
        st.plotly_chart(get_time_series_forecast(filtered_df), use_container_width=True)
        st.caption("Forecast model projects upcoming 4-week window using historical variance arrays and moving standard deviation bounds.")

    # 3. Micro Grid
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(get_time_trend(filtered_df), use_container_width=True)
        st.info(generate_insights(filtered_df, "time"))
        
        st.plotly_chart(get_severity_by_weather(filtered_df), use_container_width=True)
    with c2:
        st.plotly_chart(get_hourly_trend(filtered_df), use_container_width=True)
        st.info(generate_insights(filtered_df, "hour"))
        
        st.plotly_chart(get_causes_pie(filtered_df), use_container_width=True)

# --- TAB 2: SPATIAL INTELLIGENCE ---
with tab2:
    st.markdown("### Geospatial Hotspots & DBSCAN Clustering", help="Toggle DBSCAN Machine Learning to mathematically group geometric anomalies on the map grid.")
    enable_dbscan = st.toggle("Enable DBSCAN Neural Clustering Overlay", value=True)
    
    if len(filtered_df) == 0:
        st.warning("⚠️ No data points available for the selected filters to render the map.")
    else:
        with st.spinner("Rendering geospatial matrices..."):
            center_lat = filtered_df['Latitude'].mean()
            center_lon = filtered_df['Longitude'].mean()
            
            m = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles="CartoDB dark_matter")
            
            # Base Heatmap
            heat_data = [[row['Latitude'], row['Longitude']] for index, row in filtered_df.iterrows()]
            HeatMap(heat_data, radius=12, blur=15, name="Density Heatmap").add_to(m)
            
            # Base Markers (Clustered to prevent browser lag)
            marker_cluster = MarkerCluster(name="Accident Locations").add_to(m)
            sample_df = filtered_df.head(1000) # Process max 1000 points to keep Streamlit snappy
            for _, row in sample_df.iterrows():
                folium.CircleMarker(
                    location=[row['Latitude'], row['Longitude']],
                    radius=3,
                    color="gray",
                    fill=True,
                    popup=f"Severity: {row['Severity']}<br>Cause: {row['Cause']}"
                ).add_to(marker_cluster)
            
            # Optional DBSCAN Overlay for severe clusters
            if enable_dbscan:
                map_pts = perform_dbscan_clustering(filtered_df[filtered_df['Severity'] == 'High'].copy())
                map_pts = map_pts[map_pts['Cluster_ID'] != -1].head(1000)
                colors = ['#ff7b72', '#58a6ff', '#3fb950', '#d2a8ff', '#f0883e', '#e34c26']
                for _, row in map_pts.iterrows():
                    c_color = colors[int(row['Cluster_ID']) % len(colors)]
                    folium.CircleMarker(
                        location=[row['Latitude'], row['Longitude']], 
                        radius=8, 
                        color=c_color, 
                        fill=True,
                        fill_opacity=0.8,
                        weight=2,
                        popup=f"<b>DBSCAN Zone {row['Cluster_ID']}</b><br>Cause: {row['Cause']}"
                    ).add_to(m)
                    
            folium.LayerControl().add_to(m)
            
            # Failsafe rendering: bypass st_folium and inject raw HTML iframe
            components.html(m._repr_html_(), height=600)

# --- TAB 3: SCENARIO SIMULATOR ---
with tab3:
    st.markdown("### 🎛️ Live Probability Simulator & Impact Analysis")
    st.caption("Change infrastructure / environmental variables side-by-side to immediately evaluate the projected Risk Delta.")
    
    colA, colB, colC = st.columns([1, 1, 1.5])
    input_city = selected_city if selected_city != "All" else acc_df['City'].unique()[0]
    
    with colA:
        st.markdown("<div style='background:rgba(33,38,45,0.4); padding:20px; border-radius:10px; border:1px dashed #484f58;'>", unsafe_allow_html=True)
        st.markdown("#### Scenario A (Baseline)")
        a_hour = st.slider("Time (24h)", 0, 23, 18, key='a_hr', help="Hour of the day")
        a_weather = st.selectbox("Weather", acc_df['Weather'].unique(), index=2, key='a_we') 
        a_road = st.selectbox("Road Type", acc_df['Road_Type'].unique(), key='a_rd')
        a_lighting = st.selectbox("Lighting", acc_df['Lighting'].unique(), index=2, key='a_li') 
        a_traffic = st.selectbox("Traffic Density", acc_df['Traffic_Density'].unique(), index=2, key='a_tr') 
        st.markdown("</div>", unsafe_allow_html=True)
        
    with colB:
        st.markdown("<div style='background:rgba(33,38,45,0.4); padding:20px; border-radius:10px; border:1px dashed #484f58;'>", unsafe_allow_html=True)
        st.markdown("#### Scenario B (Proposed)")
        b_hour = st.slider("Time (24h)", 0, 23, 18, key='b_hr')
        b_weather = st.selectbox("Weather", acc_df['Weather'].unique(), index=0, key='b_we') 
        b_road = st.selectbox("Road Type", acc_df['Road_Type'].unique(), key='b_rd')
        b_lighting = st.selectbox("Lighting", acc_df['Lighting'].unique(), index=0, key='b_li') 
        b_traffic = st.selectbox("Traffic Density", acc_df['Traffic_Density'].unique(), index=1, key='b_tr')
        st.markdown("</div>", unsafe_allow_html=True)
        
    # Calculate Results
    with st.spinner("Processing Matrix through Random Forest..."):
        sA, cA, confA, XA = predict_risk_score({
            "City": input_city, "Hour": a_hour, "Time": f"{a_hour:02d}:00", "Date": "2023-01-01",
            "Weather": a_weather, "Road_Type": a_road, "Lighting": a_lighting, "Traffic_Density": a_traffic
        })
        sB, cB, confB, XB = predict_risk_score({
            "City": input_city, "Hour": b_hour, "Time": f"{b_hour:02d}:00", "Date": "2023-01-01",
            "Weather": b_weather, "Road_Type": b_road, "Lighting": b_lighting, "Traffic_Density": b_traffic
        })
        
    with colC:
        st.markdown("<div style='padding-left:20px;'>", unsafe_allow_html=True)
        diff = sB - sA
        diff_color = "#3fb950" if diff < 0 else "#ff7b72"
        st.markdown(f"<h2 style='margin:0;color:{diff_color}'>Score Delta: {diff:+.2f}%</h2>", unsafe_allow_html=True)
        st.caption(f"Expected {'reduction' if diff <0 else 'increase'} in overall risk profile upon committing to Scenario B.")
        
        g1, g2 = st.columns(2)
        with g1: st.plotly_chart(get_gauge_chart(sA, confA), use_container_width=True)
        with g2: st.plotly_chart(get_gauge_chart(sB, confB), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# --- TAB 4: BLACKSPOTS & INFRASTRUCTURE ---
with tab4:
    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.markdown("### Top Topography Blackspots")
        blackspots = get_top_blackspots(filtered_df)
        for i, row in blackspots.iterrows():
            render_blackspot_card(i+1, row['City'], row['Total_Accidents'], row['High_Severity'], row['Danger_Score'], row['Top_Cause'])
            
    with col2:
        st.markdown(f"### Intelligent Operations ({input_city})")
        insights = aggregated_recommendations(acc_df, input_city)
        if not insights:
            st.info("No immediate operational interventions flagged.")
        for (pri, title, desc, icon) in insights:
            render_recommendation_card(pri, title, desc, icon)
            
        st.markdown("### NLP Sentiment Tracker")
        sent_summary = get_city_sentiment_summary(soc_df, selected_city if selected_city != "All" else None)
        st.dataframe(sent_summary, use_container_width=True)

# --- TAB 5: EXPLAINABLE AI ---
with tab5:
    st.markdown("### Machine Learning Introspection (SHAP)")
    c1, c2 = st.columns([1, 1.5])
    
    with c1:
        st.markdown("#### Global Feature Permutation")
        st.dataframe(get_feature_importances(), use_container_width=True, hide_index=True)
        
    with c2:
        st.markdown("#### Scenario A Auto-Diagnostic")
        with st.spinner("Extracting localized SHAP values..."):
            if 'XA' in locals() and XA is not None:
                fig, explanations = explain_prediction(XA)
                for ex in explanations: st.markdown(ex)
                if fig: st.pyplot(fig)
