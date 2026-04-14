import streamlit as st

def inject_premium_css():
    '''Injects the main V4 Glassmorphic Dark Theme overrides'''
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
        
        .stApp {
            background: linear-gradient(135deg, #0d1117 0%, #161b22 100%);
            color: #c9d1d9;
            font-family: 'Inter', sans-serif;
        }
        
        /* Typography overrides */
        h1, h2, h3, h4, h5, h6 { font-family: 'Inter', sans-serif; color: #ffffff; }
        
        /* Custom Header styling */
        .brand-header {
            display: flex;
            align-items: center;
            gap: 15px;
            padding-bottom: 5px;
            border-bottom: 1px solid rgba(88, 166, 255, 0.2);
            margin-bottom: 25px;
            margin-top: -30px;
        }
        .brand-title { font-weight: 800; font-size: 34px; margin: 0; background: -webkit-linear-gradient(45deg, #58a6ff, #00f2fe); -webkit-background-clip: text; -webkit-text-fill-color: transparent;}
        .brand-subtitle { font-size: 14px; color: #8b949e; letter-spacing: 0.5px;}
        
        /* Advanced KPI Cards */
        .glass-card {
            background: rgba(33, 38, 45, 0.4);
            border: 1px solid rgba(48, 54, 61, 0.8);
            border-radius: 12px;
            padding: 24px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.4);
            text-align: center;
            backdrop-filter: blur(8px);
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }
        .glass-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 25px rgba(88, 166, 255, 0.15);
            border: 1px solid rgba(88, 166, 255, 0.4);
        }
        .metric-icon { font-size: 26px; margin-bottom: 5px;}
        .metric-value { font-size: 38px; font-weight: 800; line-height: 1.1; margin: 5px 0;}
        .metric-label { font-size: 13px; color: #8b949e; text-transform: uppercase; font-weight: 600; letter-spacing: 1px;}
        
        /* Danger Zone / Blackspot Cards */
        .blackspot-card {
            background: linear-gradient(145deg, #1c2128, #2d333b);
            border-left: 5px solid #ff7b72;
            padding: 15px 20px;
            border-radius: 8px;
            margin-bottom: 12px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            box-shadow: 0 2px 10px rgba(0,0,0,0.3);
            transition: transform 0.2s;
        }
        .blackspot-card:hover { transform: scale(1.02); }
        .bs-city { font-weight: 700; font-size: 18px; color: #fff;}
        .bs-stats { font-size: 13px; color: #8b949e;}
        
        /* Recommendation Panels */
        .rec-priority-high { border-left: 4px solid #ff7b72 !important; background: rgba(255,123,114,0.05) !important; }
        .rec-priority-med { border-left: 4px solid #d2a8ff !important; background: rgba(210,168,255,0.05) !important;}
        .rec-priority-low { border-left: 4px solid #58a6ff !important; background: rgba(88,166,255,0.05) !important;}
        .rec-panel {
            padding: 18px;
            border-radius: 6px;
            margin-bottom: 15px;
            border: 1px solid #30363d;
        }
        
    </style>
    """, unsafe_allow_html=True)

def render_header():
    st.markdown("""
    <div class="brand-header">
        <h1 style="margin:0;">🌐</h1>
        <div>
            <h1 class="brand-title">CityBrain AI</h1>
            <div class="brand-subtitle">Smart Traffic Intelligence & Analytics Platform</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_kpi(label, value, icon, color_hex="#58a6ff"):
    st.markdown(f"""
    <div class="glass-card">
        <div class="metric-icon">{icon}</div>
        <div class="metric-value" style="color: {color_hex};">{value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)

def render_blackspot_card(rank, city, accidents, severity, danger_score, cause):
    badge_color = "#ff7b72" if danger_score > 300 else "#d2a8ff"
    icon = '🔴' if danger_score > 300 else '🟠'
    
    st.markdown(f"""
    <div class="blackspot-card" style="border-left-color: {badge_color};">
        <div>
            <div class="bs-city">#{rank} {city} {icon}</div>
            <div class="bs-stats">Primary Friction: {cause}</div>
        </div>
        <div style="text-align: right;">
            <div style="font-weight: 800; font-size: 20px; color:{badge_color};">{danger_score}</div>
            <div class="bs-stats" style="font-size:11px;">Danger Index ({accidents} total)</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_recommendation_card(priority, title, desc, icon):
    p_class = "rec-priority-low"
    if priority == "HIGH": p_class = "rec-priority-high"
    elif priority == "MEDIUM": p_class = "rec-priority-med"
    
    st.markdown(f"""
    <div class="rec-panel {p_class}">
        <div style="font-weight:700; margin-bottom:5px; color:#fff;">{icon} {title}  <span style="font-size:10px; padding:2px 6px; border-radius:10px; background:rgba(255,255,255,0.1); margin-left:10px;">{priority} PRIORITY</span></div>
        <div style="color:#8b949e; font-size:14px; line-height:1.4;">{desc}</div>
    </div>
    """, unsafe_allow_html=True)
