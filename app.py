import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import plotly.graph_objects as go

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Cardio Predict AI",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==========================================
# 2. SESSION STATE & THEME LOGIC
# ==========================================
if 'page' not in st.session_state:
    st.session_state['page'] = 'dashboard'
if 'theme' not in st.session_state:
    st.session_state['theme'] = 'light'

def toggle_theme():
    if st.session_state['theme'] == 'light':
        st.session_state['theme'] = 'dark'
    else:
        st.session_state['theme'] = 'light'

def set_page(page_name):
    st.session_state['page'] = page_name

# Define Theme Colors
if st.session_state['theme'] == 'light':
    bg_gradient = "linear-gradient(135deg, #fdfbfb 0%, #ebedee 100%)" # Clean Grey/White
    
    # Deep Blue to Aqua (User preferred)
    navbar_bg = "linear-gradient(135deg, #1e3a8a, #22d3ee)" 
    
    text_color = "#2c3e50"
    card_bg = "rgba(255, 255, 255, 0.95)"
    card_shadow = "rgba(0,0,0,0.08)"
else:
    bg_gradient = "linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%)" # Deep Ocean
    navbar_bg = "linear-gradient(135deg, #1e3a8a, #22d3ee)" #90deg, #2C3E50 0%, #4CA1AF 100%)" # Dark Slate to Teal
    text_color = "#e0e0e0";
    card_bg = "rgba(30, 40, 50, 0.85)"
    card_shadow = "rgba(0,0,0,0.4)"

# Custom CSS Injection
st.markdown(f"""
<style>
    /* Global Styles */
    body {{
        font-family: 'Roboto', sans-serif;
    }}
    
    /* Main Background */
    .stApp {{
        background: {bg_gradient};
        color: {text_color};
    }}
    
    /* Text Color Override for Streamlit elements */
    h1, h2, h3, h4, h5, h6, p, label {{
        color: {text_color} !important;
    }}
    
    /* Navbar Styling */
    .navbar-container {{
        background: {navbar_bg};
        padding: 15px 25px;
        border-radius: 12px;
        margin-bottom: 25px;
        box-shadow: 0 4px 6px {card_shadow};
        display: flex;
        justify-content: space-between;
        align-items: center;
    }}
    
    /* Button Styling */
    div.stButton > button {{
        background: {navbar_bg};
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        font-size: 0.95rem;
        font-weight: 500;
        transition: all 0.3s ease;
        width: 180px;
        height: 44px;
        display: flex;
        gap: 6px;
        margin:auto;
        font-color:white;
        align-items: center;  
        justify-content: center; 
        transition: all 0.3s ease;
    }}
    div.stButton > button:hover {{
        filter: brightness(1.1);
        transform: translateY(-2px);
        box-shadow: 0 4px 8px {card_shadow};
    }}
    
    /* Card Styling */
    .card {{
        background-color: {card_bg};
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 15px {card_shadow};
        margin-bottom: 20px;
        border-left: 5px solid #4b6cb7;
        transition: transform 0.2s;
    }}
    .card:hover {{
        transform: translateY(-5px);
    }}
    .metric-value {{
        font-size: 2.2rem;
        font-weight: 700;
        color: {text_color};
        margin-bottom: 5px;
    }}
    .metric-label {{
        color: {text_color};
        opacity: 0.8;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }}
    
    /* Form Inputs */
    .stTextInput>div>div>input, .stNumberInput>div>div>input, .stSelectbox>div>div>div {{
        border-radius: 8px;
        border: 1px solid rgba(0,0,0,0.1);
    }}
    
    /* Charts */
    .stPyplot {{
        background-color: {card_bg};
        padding: 15px;
        border-radius: 15px;
    }}

    hr {{
        border-color: rgba(255,255,255,0.2);
    }}
</style>
""", unsafe_allow_html=True)

# Extended CSS for Animations & Layout
st.markdown(f"""
<style>
    /* Animations */
    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(20px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    
    .card {{
        animation: fadeIn 0.8s ease-out;
    }}
    
    /* Centered Title */
    .main-title {{
        text-align: center;
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #1A2980, #26D0CE);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
    }}
    .subtitle {{
        text-align: center;
        color: {text_color};
        font-size: 1.2rem;
        margin-bottom: 30px;
        opacity: 0.8;
    }}
    
    /* Bigger Layout / Cards */
    .card {{
        padding: 30px; /* Bigger padding */
        border-radius: 20px; /* Softer rounded corners */
        background-color: {card_bg};
        box-shadow: 0 10px 25px {card_shadow}; /* Deeper shadow */
        margin-bottom: 30px;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }}
    .card:hover {{
        transform: translateY(-8px) scale(1.01);
        box-shadow: 0 15px 35px {card_shadow};
    }}
    
    /* Navbar Buttons Centering */
    .stButton > button {{
        width: 100%;
        font-size: 1.1rem;
        padding: 0.8rem 1.5rem;
    }}
    
    /* Target Streamlit Form to look like a card */
    [data-testid="stForm"] {{
        padding: 30px;
        border-radius: 20px;
        background-color: {card_bg};
        box-shadow: 0 10px 25px {card_shadow};
        border-left: 5px solid #4b6cb7;
        animation: fadeIn 0.8s ease-out;
    }}
    
    /* Target Streamlit Plots to look like cards */
    [data-testid="stPyplot"] {{
        padding: 20px;
        border-radius: 20px;
        background-color: {card_bg};
        box-shadow: 0 10px 25px {card_shadow};
        border-left: 5px solid #4b6cb7;
        margin-bottom: 20px;
        animation: fadeIn 0.8s ease-out;
    }}
</style>
""", unsafe_allow_html=True)

st.markdown(f"""
<style>
/* Disclaimer Card Styling - Blue Theme Matching Your App */
.disclaimer-card {{
    background: {card_bg};
    color: {text_color};
    padding: 28px;
    border-radius: 20px;
    box-shadow: 0 10px 25px {card_shadow};
    margin-bottom: 25px;
    border-left: 6px solid #4b6cb7; 
    animation: fadeIn 0.8s ease-out;
}}

.disclaimer-card h2 {{
    margin-top: 0;
    font-size: 1.4rem;
    font-weight: 700;
    background: linear-gradient(90deg, #1A2980, #26D0CE);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}}

.disclaimer-card p,
.disclaimer-card li {{
    font-size: 1rem;
    line-height: 1.7;
    opacity: 0.95;
}}

.disclaimer-card ul {{
    padding-left: 20px;
}}

.disclaimer-card li {{
    margin-bottom: 8px;
}}

/* Hover effect */
.disclaimer-card:hover {{
    transform: translateY(-5px);
    box-shadow: 0 15px 35px {card_shadow};
    transition: all 0.3s ease;
}}
</style>
""", unsafe_allow_html=True)


# ==========================================
# 3. TOP NAVIGATION BAR
# ==========================================

# 1. Centered Title
st.markdown('<h1 class="main-title">🫀 Cardio Predict AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Advanced Cardiovascular Risk Assessment System</p>', unsafe_allow_html=True)

# 2. Navbar Row (Centered)
# We use columns to center the buttons. 
# Layout: [Spacer] [Dash] [Pred] [Perf] [Docs] [Spacer] [Theme]
# _, c1, c2, c3, c4,c5 ,_, c_theme = st.columns([1, 2, 2, 2, 2 ,2, 0.5, 1])
c1, c2, c3, c4, c5, c_theme = st.columns(6)

with c1:
    if st.button("📊 Dashboard"): set_page('dashboard')
with c2:
    if st.button("  🩺 Predict   "): set_page('prediction')
with c3:
    if st.button("📈 Performance"): set_page('performance')
with c4:
    if st.button("📖 About"): set_page('About')
with c5:
    if st.button("⚠️ Disclaimer"): set_page('Disclaimer')
with c_theme:
    if st.button("🌓 Mode"):
        toggle_theme()
        st.rerun()

st.markdown("---")

# ==========================================
# 4. LOAD RESOURCES
# ==========================================

model = joblib.load("lr_model.pkl")  
scaler = joblib.load("scaler.pkl")

models = {
    "lr": model,
    "scaler": scaler
}

    # Load Data (Only Exploring_Dataset.csv)
# =========================
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("Exploring_Dataset.csv")
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

df = load_data()

# ==========================================
# 5. PAGE CONTENT
# ==========================================

# --- DASHBOARD ---
if st.session_state['page'] == 'dashboard':
    st.markdown("### Medical Analytics Dashboard")
    
    if df is not None:
        # Key Metrics
        m1, m2, m3, m4 = st.columns(4)
        
        total = len(df)
        pos = df['cardio'].sum() if 'cardio' in df.columns else 0
        neg = total - pos
        accuracy = 72  # percent
        avg_age = (df['age'].mean() / 365.25) if 'age' in df.columns else 0
        
        with m1:
            st.markdown(f'<div class="card"><div class="metric-label">Patient Records</div><div class="metric-value">{total:,}</div></div>', unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div class="card"><div class="metric-label">Positive Cases</div><div class="metric-value" style="color: #ef5350;">{pos:,}</div></div>', unsafe_allow_html=True)
        with m3:
            st.markdown(f'<div class="card"><div class="metric-label">Negative Cases</div><div class="metric-value" style="color: #66bb6a;">{neg:,}</div></div>', unsafe_allow_html=True)
        with m4:
            st.markdown(f'<div class="card"><div class="metric-label">Model Accuracy</div><div class="metric-value">{accuracy:,}%</div></div>', unsafe_allow_html=True)
            
        # Charts
        c1, c2 = st.columns(2)
        
        # Theme-aware plotting
        if st.session_state['theme'] == 'dark':
            plt.style.use("dark_background")
        else:
             # Use seaborn style via sns which handles versions better, or a safe matplotlib one
            sns.set_style("whitegrid")
            # plt.style.use("seaborn-v0_8-whitegrid") # Safe alternative if needed

        
        with c1:
            st.markdown("#### Distribution of Age")
            fig1, ax1 = plt.subplots(figsize=(6, 4))
            fig1.patch.set_alpha(0)
            ax1.patch.set_alpha(0)
            if 'age' in df.columns:
                sns.histplot(df['age']/365.25, bins=30, kde=True, color='#26c6da', ax=ax1)
                ax1.set_xlabel("Age (Years)", color=text_color)
                ax1.set_ylabel("Count", color=text_color)
                ax1.tick_params(colors=text_color)
                for spine in ax1.spines.values(): spine.set_color(text_color)
                st.pyplot(fig1)
            
        with c2:
            st.markdown("#### Patient Gender Split")
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            fig2.patch.set_alpha(0)
            ax2.patch.set_alpha(0)
            if 'gender' in df.columns:
                sns.countplot(x=df['gender'], palette="viridis", ax=ax2)
                ax2.set_xticklabels(['Female', 'Male'])
                ax2.set_xlabel("Gender", color=text_color)
                ax2.set_ylabel("Count", color=text_color)
                ax2.tick_params(colors=text_color)
                for spine in ax2.spines.values(): spine.set_color(text_color)
                st.pyplot(fig2)
    else:
        st.error("Dataset 'Exploring_Dataset.csv' not found. Please upload it to the project directory.")



# --- PREDICTION ---
elif st.session_state['page'] == 'prediction':
    st.markdown("<h3>🔮 Patient Health Assessment</h3>", unsafe_allow_html=True)
    st.warning("⚠️ For educational purposes only. Not for medical diagnosis.")
    
    with st.form("pred"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("#### 👤 Demographics")
            age_in_years = st.number_input("Age (Years)", 18, 100, 50)
            gender = st.selectbox("Gender", [1, 2], format_func=lambda x: "Female" if x==1 else "Male")
            height = st.number_input("Height (cm)", 140, 250, 165)
            weight = st.number_input("Weight (kg)", 40, 200, 70)
        with c2:
            st.markdown("#### 🩺 Clinical")
            ap_hi = st.number_input("Systolic BP", 90, 250, 120)
            ap_lo = st.number_input("Diastolic BP", 60, 150, 80)
            cholesterol = st.selectbox("Cholesterol", [1,2,3], format_func=lambda x: ["Normal","Above Normal","High"][x-1])
            gluc = st.selectbox("Glucose", [1,2,3], format_func=lambda x: ["Normal","Above Normal","High"][x-1])
        with c3:
            st.markdown("#### 🏃 Lifestyle")
            # Using 0/1 for inputs to match model expectation
            smoke = st.selectbox("Smoking", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
            alco = st.selectbox("Alcohol", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
            active = st.selectbox("Activity", [0,1], format_func=lambda x: "Inactive" if x==0 else "Active")
            
            #Calculate BMI for display
            bmi = weight / ((height / 100) ** 2)

        # No RF/GB Selection anymore
        st.markdown("**Model**: Logistic Regression (Default)")

        submit = st.form_submit_button("🔍 Analyze", use_container_width=True)
    
    if submit:
        # Prepare input data in the exact order as training
        # Features: age (days), gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active
        # NOTE: Model does NOT use BMI. Model uses Age in DAYS.
        user_data = np.array([[gender, height, weight, ap_hi, ap_lo, cholesterol, 
                              gluc, smoke, alco, active, age_in_years, bmi]])
        
        # Scale the data
        user_data_scaled = scaler.transform(user_data)
                
        # Predict
        model = models.get('lr')
        if model:
            try:
                prediction = model.predict(user_data_scaled)[0]
                probability = model.predict_proba(user_data_scaled)[0][1]
                
                # User Requested CSS for Results
                st.markdown("""
                <style>
                .risk-high {
                    background: linear-gradient(135deg, #dc2626, #b91c1c);
                    padding: 25px;
                    border-radius: 12px;
                    color: white;
                    text-align: center;
                    font-size: 24px;
                    font-weight: 600;
                    margin-bottom: 20px;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.2);
                }
                .risk-low {
                    background: linear-gradient(135deg, #059669, #047857);
                    padding: 25px;
                    border-radius: 12px;
                    color: white;
                    text-align: center;
                    font-size: 24px;
                    font-weight: 600;
                    margin-bottom: 20px;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.2);
                }
                .info-box {
                    background-color: rgba(255,255,255,0.9);
                    padding: 18px;
                    border-radius: 10px;
                    border-left: 5px solid #0891b2;
                    font-size: 16px;
                    color: #1e3a5f;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                }
                </style>
                """, unsafe_allow_html=True)
                
                # Display results
                st.markdown("<br>", unsafe_allow_html=True)
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    if prediction == 1:
                        st.markdown(f"""
                        <div class='risk-high'>
                            ⚠️ HIGH RISK of Cardiovascular Disease<br>
                            <span style='font-size: 1rem;'>Risk Probability: {probability:.1%}</span>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown("""
                        <div class='info-box' style='border-left-color: #ff6b6b; margin-top: 20px;'>
                            <strong>⚠️ Recommended Actions:</strong><br><br>
                            <strong style='color: #ff6b6b;'>⚠️ IMPORTANT: Consult a healthcare professional immediately for proper medical evaluation.</strong><br><br>
                            General recommendations may include:<br>
                            • Schedule an appointment with a cardiologist as soon as possible<br>
                            • Monitor blood pressure daily and keep a log<br>
                            • Consider lifestyle modifications (diet, exercise, stress management)<br>
                            • Avoid smoking and limit alcohol consumption<br>
                            • Follow all prescribed medication regimens carefully<br>
                            • Get regular health checkups and cardiovascular screenings<br><br>
                            <em style='color: #9ca3af; font-size: 0.85rem;'>Remember: This is an AI prediction, not a medical diagnosis.</em>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class='risk-low'>
                            ✓ LOW RISK of Cardiovascular Disease<br>
                            <span style='font-size: 1rem;'>Risk Probability: {probability:.1%}</span>
                        </div>
                        """, unsafe_allow_html=True)
                       
                        st.markdown("""
                        <div class='info-box' style='border-left-color: #51cf66; margin-top: 20px;'>
                            <strong>✓ Maintenance Recommendations:</strong><br><br>
                            <strong style='color: #51cf66;'>✓ Good news! However, regular medical checkups are still essential.</strong><br><br>
                            General health maintenance tips:<br>
                            • Continue maintaining a healthy lifestyle<br>
                            • Regular physical activity (150 minutes per week recommended)<br>
                            • Balanced diet rich in fruits, vegetables, and whole grains<br>
                            • Annual cardiovascular health checkups<br>
                            • Monitor blood pressure periodically<br>
                            • Stay hydrated and manage stress effectively<br><br>
                            <em style='color: #9ca3af; font-size: 0.85rem;'>Note: Low risk does not mean zero risk. Continue medical supervision.</em>
                        </div>
                        """, unsafe_allow_html=True)
                
                # # Feature contribution chart
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("### 📊 Risk Factor Contribution Analysis")
                    
                feature_names = ['Age', 'Gender', 'Height', 'Weight', 'Systolic BP', 'Diastolic BP', 
                                'Cholesterol', 'Glucose', 'Smoking', 'Alcohol', 'Activity','BMI']
                
                if hasattr(model, 'coef_'):
                    w = model.coef_[0]
                    contributions = user_data_scaled[0] * w.flatten()
                    
                    # Ensure matching length
                    if len(contributions) == len(feature_names):
                        contrib_df = pd.DataFrame({
                            'Feature': feature_names,
                            'Contribution': contributions
                        }).sort_values('Contribution', key=abs, ascending=False)
                        
                        fig_contrib = go.Figure()
                        colors = ['#dc2626' if c > 0 else '#059669' for c in contrib_df['Contribution']]
                        fig_contrib.add_trace(go.Bar(
                            y=contrib_df['Feature'],
                            x=contrib_df['Contribution'],
                            orientation='h',
                            marker_color=colors,
                            text=[f"{c:+.3f}" for c in contrib_df['Contribution']],
                            textposition='outside'
                        ))
                        fig_contrib.update_layout(
                            xaxis_title="Impact on Risk (Positive = Increases Risk)",
                            yaxis_title="",
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='#334155'),
                            height=400,
                            showlegend=False
                        )
                        st.plotly_chart(fig_contrib, use_container_width=True)
                    else:
                        st.warning("Feature contribution count mismatch.")
                st.markdown("<br>", unsafe_allow_html=True)
                
                st.markdown("""
                <div class='info-box'>
                    <strong>ℹ️ Understanding This Chart:</strong><br>
                    This chart shows how each factor contributes to the overall risk prediction. 
                    Red bars increase risk (push toward disease), while green bars decrease risk (push toward health). 
                    Longer bars have stronger influence on the prediction.<br><br>
                    <em style='color: #ff6b6b; font-size: 0.85rem;'>⚠️ Disclaimer: This analysis is for educational purposes only and should not replace professional medical assessment.</em>
                </div>
                """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Prediction logic error: {e}")
        else:
             st.error("Model not loaded.")


# --- PERFORMANCE PAGE ---
elif st.session_state['page'] == 'performance':
    st.markdown('<h1 style="text-align:center; color:#1A2980;">📈 Model Performance Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align:center; opacity:0.8;">Comprehensive Analysis of Logistic Regression Model</p>', unsafe_allow_html=True)
    st.markdown("---")

    model = models.get('lr', None)
    if model is None:
        st.error("❌ Model not loaded. Cannot display performance metrics.")
    else:
        # ===== MODEL OVERVIEW CARDS =====
        st.markdown("### 📊 Model Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="card" style="text-align:center;">
                <div class="metric-label">Model Type</div>
                <div class="metric-value" style="font-size:1.5rem;">LR</div>
                <p style="font-size:0.8rem; opacity:0.7;">Logistic Regression</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="card" style="text-align:center;">
                <div class="metric-label">Features</div>
                <div class="metric-value" style="font-size:1.5rem;">12</div>
                <p style="font-size:0.8rem; opacity:0.7;">Input Variables</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="card" style="text-align:center;">
                <div class="metric-label">Classes</div>
                <div class="metric-value" style="font-size:1.5rem;">2</div>
                <p style="font-size:0.8rem; opacity:0.7;">Binary Classification</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="card" style="text-align:center;">
                <div class="metric-label">Algorithm</div>
                <div class="metric-value" style="font-size:1.5rem;">✓</div>
                <p style="font-size:0.8rem; opacity:0.7;">Interpretable</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ===== FEATURE IMPORTANCE =====
        if hasattr(model, 'coef_'):
            st.markdown("### 🎯 Feature Importance Analysis")
            st.caption("Understanding which features drive cardiovascular risk predictions")
            
            feats = ['Age', 'Gender', 'Height', 'Weight', 'Systolic BP', 'Diastolic BP', 
                     'Cholesterol', 'Glucose', 'Smoking', 'Alcohol', 'Activity', 'BMI']
            coefs = model.coef_[0]
            
            if len(coefs) == len(feats):
                coef_df = pd.DataFrame({
                    'Feature': feats, 
                    'Coefficient': coefs,
                    'Abs_Coef': np.abs(coefs)
                }).sort_values('Coefficient', ascending=True)
                
                col1, col2 = st.columns([3, 2])
                
                with col1:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    fig_imp, ax_imp = plt.subplots(figsize=(8, 6))
                    fig_imp.patch.set_alpha(0)
                    ax_imp.patch.set_alpha(0)
                    
                    colors = ['#059669' if c < 0 else '#dc2626' for c in coef_df['Coefficient']]
                    bars = ax_imp.barh(coef_df['Feature'], coef_df['Coefficient'], color=colors, alpha=0.8)
                    
                    ax_imp.axvline(0, color=text_color, linestyle='--', alpha=0.5, linewidth=2)
                    ax_imp.set_xlabel("Coefficient Value (Impact on Risk)", color=text_color, fontsize=11, fontweight='bold')
                    ax_imp.set_ylabel("")
                    ax_imp.set_title("Feature Coefficients", color=text_color, fontsize=13, fontweight='bold', pad=15)
                    ax_imp.tick_params(colors=text_color, labelsize=10)
                    ax_imp.grid(axis='x', alpha=0.3, linestyle=':', color=text_color)
                    
                    for spine in ax_imp.spines.values():
                        spine.set_color(text_color)
                        spine.set_linewidth(1.5)
                    
                    # Add value labels
                    for bar in bars:
                        width = bar.get_width()
                        label_x = width + (0.02 if width > 0 else -0.02)
                        ax_imp.text(label_x, bar.get_y() + bar.get_height()/2, 
                                   f'{width:.3f}',
                                   ha='left' if width > 0 else 'right',
                                   va='center', fontsize=8, color=text_color, fontweight='bold')
                    
                    plt.tight_layout()
                    st.pyplot(fig_imp)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.markdown("#### 📖 Interpretation Guide")
                    st.markdown("""
                    **Positive Coefficients (Red):**
                    - Increase cardiovascular risk
                    - Higher values = Higher risk
                    
                    **Negative Coefficients (Green):**
                    - Decrease cardiovascular risk
                    - Higher values = Lower risk
                    
                    **Magnitude Matters:**
                    - Larger absolute values = Stronger impact
                    - Features closer to zero have minimal effect
                    """)
                    
                    # Top risk factors
                    top_risk = coef_df.nlargest(3, 'Abs_Coef')
                    st.markdown("**🔴 Top Risk Factors:**")
                    for idx, row in top_risk.iterrows():
                        direction = "↑ Increases" if row['Coefficient'] > 0 else "↓ Decreases"
                        st.markdown(f"- **{row['Feature']}**: {direction} risk")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.warning(f"⚠️ Feature count mismatch: model has {len(coefs)}, expected {len(feats)}.")
        
        st.markdown("<br>", unsafe_allow_html=True)

       # ===== PERFORMANCE METRICS =====
        if not df.empty and 'cardio' in df.columns:
            X_cols = ['gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 
                     'gluc', 'smoke', 'alco', 'active', 'age', 'bmi']
            
            if all(c in df.columns for c in X_cols):
                X_sample = df[X_cols].values
                y_sample = df['cardio'].values
                
                try:
                    X_scaled = scaler.transform(X_sample)
                except:
                    X_scaled = X_sample
                
                y_pred = model.predict(X_scaled)
                y_prob = model.predict_proba(X_scaled)[:, 1]

                from sklearn.metrics import (confusion_matrix, classification_report, 
                                            roc_curve, auc, accuracy_score, 
                                            precision_score, recall_score, f1_score)

                # Calculate metrics
                accuracy = accuracy_score(y_sample, y_pred)
                precision = precision_score(y_sample, y_pred)
                recall = recall_score(y_sample, y_pred)
                f1 = f1_score(y_sample, y_pred)

                # ===== METRICS CARDS =====
                st.markdown("### 📊 Performance Metrics")
                m1, m2, m3, m4 = st.columns(4)
                
                with m1:
                    st.markdown(f"""
                    <div class="card" style="text-align:center; border-left-color:#1A2980;">
                        <div class="metric-label">Accuracy</div>
                        <div class="metric-value" style="color:#1A2980;">{accuracy:.1%}</div>
                        <p style="font-size:0.75rem; opacity:0.7;">Overall Correctness</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with m2:
                    st.markdown(f"""
                    <div class="card" style="text-align:center; border-left-color:#26D0CE;">
                        <div class="metric-label">Precision</div>
                        <div class="metric-value" style="color:#26D0CE;">{precision:.1%}</div>
                        <p style="font-size:0.75rem; opacity:0.7;">Positive Predictive Value</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with m3:
                    st.markdown(f"""
                    <div class="card" style="text-align:center; border-left-color:#059669;">
                        <div class="metric-label">Recall</div>
                        <div class="metric-value" style="color:#059669;">{recall:.1%}</div>
                        <p style="font-size:0.75rem; opacity:0.7;">Sensitivity</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with m4:
                    st.markdown(f"""
                    <div class="card" style="text-align:center; border-left-color:#dc2626;">
                        <div class="metric-label">F1-Score</div>
                        <div class="metric-value" style="color:#dc2626;">{f1:.1%}</div>
                        <p style="font-size:0.75rem; opacity:0.7;">Harmonic Mean</p>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # ===== CONFUSION MATRIX & ROC CURVE =====
                st.markdown("### 🎯 Detailed Performance Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.markdown("#### Confusion Matrix")
                    st.caption("Actual vs Predicted Classifications")
                    
                    cm = confusion_matrix(y_sample, y_pred)
                    
                    fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
                    fig_cm.patch.set_alpha(0)
                    ax_cm.patch.set_alpha(0)
                    
                    sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn_r', 
                               xticklabels=['Negative', 'Positive'],
                               yticklabels=['Negative', 'Positive'],
                               cbar_kws={'label': 'Count'},
                               ax=ax_cm, linewidths=2, linecolor=text_color)
                    
                    ax_cm.set_xlabel("Predicted Label", color=text_color, fontsize=11, fontweight='bold')
                    ax_cm.set_ylabel("True Label", color=text_color, fontsize=11, fontweight='bold')
                    ax_cm.set_title("Confusion Matrix", color=text_color, fontsize=12, fontweight='bold', pad=15)
                    ax_cm.tick_params(colors=text_color, labelsize=10)
                    
                    plt.tight_layout()
                    st.pyplot(fig_cm)
                    
                    # Add interpretation
                    tn, fp, fn, tp = cm.ravel()
                    st.markdown(f"""
                    **Matrix Breakdown:**
                    - ✅ True Negatives: {tn:,}
                    - ❌ False Positives: {fp:,}
                    - ❌ False Negatives: {fn:,}
                    - ✅ True Positives: {tp:,}
                    """)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.markdown("#### ROC Curve")
                    st.caption("Receiver Operating Characteristic")
                    
                    fpr, tpr, thresholds = roc_curve(y_sample, y_prob)
                    roc_auc = auc(fpr, tpr)
                    
                    fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
                    fig_roc.patch.set_alpha(0)
                    ax_roc.patch.set_alpha(0)
                    
                    ax_roc.plot(fpr, tpr, color='#1A2980', lw=3, 
                               label=f'ROC Curve (AUC = {roc_auc:.3f})')
                    ax_roc.plot([0, 1], [0, 1], color='gray', lw=2, 
                               linestyle='--', label='Random Classifier')
                    
                    ax_roc.fill_between(fpr, tpr, alpha=0.2, color='#26D0CE')
                    
                    ax_roc.set_xlabel("False Positive Rate", color=text_color, fontsize=11, fontweight='bold')
                    ax_roc.set_ylabel("True Positive Rate", color=text_color, fontsize=11, fontweight='bold')
                    ax_roc.set_title("ROC Curve Analysis", color=text_color, fontsize=12, fontweight='bold', pad=15)
                    ax_roc.tick_params(colors=text_color, labelsize=10)
                    ax_roc.legend(loc="lower right", fontsize=9)
                    ax_roc.grid(True, alpha=0.3, linestyle=':', color=text_color)
                    
                    for spine in ax_roc.spines.values():
                        spine.set_color(text_color)
                        spine.set_linewidth(1.5)
                    
                    plt.tight_layout()
                    st.pyplot(fig_roc)
                    
                    # AUC interpretation
                    if roc_auc >= 0.9:
                        quality = "🌟 Excellent"
                    elif roc_auc >= 0.8:
                        quality = "✅ Good"
                    elif roc_auc >= 0.7:
                        quality = "👍 Fair"
                    else:
                        quality = "⚠️ Poor"
                    
                    st.markdown(f"""
                    **AUC Score: {roc_auc:.3f}**
                    - Model Quality: {quality}
                    - Interpretation: Model's ability to distinguish between positive and negative cases
                    """)
                    st.markdown('</div>', unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                # ===== METRICS CARDS =====
                st.markdown("### 📊 Performance Metrics Across Models")
                m1, m2, m3 = st.columns(3)
                
                # Logistic Regression Accuracy
                with m1:
                    st.markdown(f"""
                    <div class="card" style="text-align:center; border-left-color:#1A2980;">
                        <div class="metric-label">Logistic Regression</div>
                        <div class="metric-value" style="color:#1A2980;">{accuracy:.1%}</div>
                        <p style="font-size:0.75rem; opacity:0.7;">Overall Accuracy</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Random Forest Accuracy
                rf_accuracy = 0.70
                with m2:
                    st.markdown(f"""
                    <div class="card" style="text-align:center; border-left-color:#F59E0B;">
                        <div class="metric-label">Random Forest</div>
                        <div class="metric-value" style="color:#F59E0B;">{rf_accuracy:.1%}</div>
                        <p style="font-size:0.75rem; opacity:0.7;">Overall Accuracy</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Gradient Boosting Accuracy
                gb_accuracy = 0.73
                with m3:
                    st.markdown(f"""
                    <div class="card" style="text-align:center; border-left-color:#6366F1;">
                        <div class="metric-label">Gradient Boosting</div>
                        <div class="metric-value" style="color:#6366F1;">{gb_accuracy:.1%}</div>
                        <p style="font-size:0.75rem; opacity:0.7;">Overall Accuracy</p>
                    </div>
                    """, unsafe_allow_html=True)


                # ===== CLASSIFICATION REPORT =====
                st.markdown("### 📋 Detailed Classification Report")
                
                report_dict = classification_report(y_sample, y_pred, 
                                                   target_names=['Negative', 'Positive'],
                                                   output_dict=True)
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    
                    # Create DataFrame for visualization
                    report_df = pd.DataFrame({
                        'Class': ['Negative (0)', 'Positive (1)'],
                        'Precision': [report_dict['Negative']['precision'], 
                                     report_dict['Positive']['precision']],
                        'Recall': [report_dict['Negative']['recall'], 
                                  report_dict['Positive']['recall']],
                        'F1-Score': [report_dict['Negative']['f1-score'], 
                                    report_dict['Positive']['f1-score']],
                        'Support': [int(report_dict['Negative']['support']), 
                                   int(report_dict['Positive']['support'])]
                    })
                    
                    fig_class, ax_class = plt.subplots(figsize=(8, 4))
                    fig_class.patch.set_alpha(0)
                    ax_class.patch.set_alpha(0)
                    
                    x = np.arange(len(report_df['Class']))
                    width = 0.25
                    
                    bars1 = ax_class.bar(x - width, report_df['Precision'], width, 
                                        label='Precision', color='#1A2980', alpha=0.8)
                    bars2 = ax_class.bar(x, report_df['Recall'], width, 
                                        label='Recall', color='#26D0CE', alpha=0.8)
                    bars3 = ax_class.bar(x + width, report_df['F1-Score'], width, 
                                        label='F1-Score', color='#059669', alpha=0.8)
                    
                    ax_class.set_xlabel("Class", color=text_color, fontsize=11, fontweight='bold')
                    ax_class.set_ylabel("Score", color=text_color, fontsize=11, fontweight='bold')
                    ax_class.set_title("Per-Class Performance Metrics", color=text_color, 
                                      fontsize=12, fontweight='bold', pad=15)
                    ax_class.set_xticks(x)
                    ax_class.set_xticklabels(report_df['Class'])
                    ax_class.tick_params(colors=text_color, labelsize=10)
                    ax_class.legend(fontsize=9)
                    ax_class.set_ylim(0, 1.1)
                    ax_class.grid(axis='y', alpha=0.3, linestyle=':', color=text_color)
                    
                    for spine in ax_class.spines.values():
                        spine.set_color(text_color)
                        spine.set_linewidth(1.5)
                    
                    plt.tight_layout()
                    st.pyplot(fig_class)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.markdown("#### 📊 Summary Statistics")
                    st.markdown(f"""
                    **Overall Metrics:**
                    - Macro Avg Precision: {report_dict['macro avg']['precision']:.3f}
                    - Macro Avg Recall: {report_dict['macro avg']['recall']:.3f}
                    - Macro Avg F1: {report_dict['macro avg']['f1-score']:.3f}
                    
                    **Weighted Metrics:**
                    - Weighted Precision: {report_dict['weighted avg']['precision']:.3f}
                    - Weighted Recall: {report_dict['weighted avg']['recall']:.3f}
                    - Weighted F1: {report_dict['weighted avg']['f1-score']:.3f}
                    
                    **Dataset:**
                    - Total Samples: {int(report_dict['Negative']['support'] + report_dict['Positive']['support']):,}
                    - Class Balance: {report_dict['Positive']['support'] / (report_dict['Negative']['support'] + report_dict['Positive']['support']):.1%} positive
                    """)
                    st.markdown('</div>', unsafe_allow_html=True)

            else:
                st.warning("⚠️ Required columns missing for performance evaluation.")
        else:
            st.info("ℹ️ Dataset or 'cardio' column missing. Cannot compute performance metrics.")

        # ===== MODEL INSIGHTS =====
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 💡 Model Insights & Recommendations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div class="card">
                <h4>✅ Strengths</h4>
                <ul>
                    <li><b>Interpretability:</b> Clear coefficient-based feature importance</li>
                    <li><b>Speed:</b> Fast training and prediction times</li>
                    <li><b>Simplicity:</b> Easy to understand and deploy</li>
                    <li><b>Probabilistic:</b> Provides confidence scores for predictions</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="card">
                <h4>⚠️ Limitations</h4>
                <ul>
                    <li><b>Linear Assumptions:</b> May miss complex non-linear patterns</li>
                    <li><b>Feature Engineering:</b> Requires careful feature selection</li>
                    <li><b>Educational Use:</b> Not validated for clinical deployment</li>
                    <li><b>Data Quality:</b> Performance depends on training data quality</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            

# --- ABOUT PAGE ---
elif st.session_state['page'] == 'About':
    st.markdown(
        '<h1 style="text-align:center; background: linear-gradient(90deg, #1A2980, #26D0CE); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">About Cardio Predict AI</h1>',
        unsafe_allow_html=True
    )
    st.markdown(
        f'<p style="text-align:center; font-size:1.2rem; font-weight:600; color:{text_color}; margin-top:-10px; margin-bottom:20px;">Made by Jagruti ✨</p>',
        unsafe_allow_html=True
    )
    st.markdown("---")

    # ==== PROJECT OVERVIEW ====
    st.markdown(f"""
    <div class="card">
        <h3>🎯 What is Cardio Predict AI?</h3>
        <p style="font-size: 1.05rem; line-height: 1.8;">
        An intelligent cardiovascular disease risk assessment system built with machine learning. 
        This educational platform demonstrates how AI can analyze clinical data to predict CVD risk 
        based on health metrics and lifestyle factors.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ==== KEY FEATURES ====
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="card" style="text-align: center;">
            <h1 style="font-size: 2.5rem;">🤖</h1>
            <h4>AI Prediction</h4>
            <p style="font-size: 0.9rem;">Logistic Regression model with ~72% accuracy on 70,000+ records</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="card" style="text-align: center;">
            <h1 style="font-size: 2.5rem;">📊</h1>
            <h4>Interactive Dashboard</h4>
            <p style="font-size: 0.9rem;">Real-time visualizations and performance analytics</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="card" style="text-align: center;">
            <h1 style="font-size: 2.5rem;">🔍</h1>
            <h4>Interpretable</h4>
            <p style="font-size: 0.9rem;">Clear explanations of predictions and feature importance</p>
        </div>
        """, unsafe_allow_html=True)

    # ==== TECHNOLOGY STACK ====
    st.markdown(f"""
    <div class="card">
        <h3>🛠️ Technology Stack</h3>
    </div>
    """, unsafe_allow_html=True)

    tech1, tech2, tech3 = st.columns(3)
    
    with tech1:
        st.markdown(f"""
        <div class="card">
            <h4>Core</h4>
            <ul style="font-size: 0.95rem;">
                <li>Python 3.x</li>
                <li>Streamlit</li>
                <li>Pandas & NumPy</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with tech2:
        st.markdown(f"""
        <div class="card">
            <h4>Machine Learning</h4>
            <ul style="font-size: 0.95rem;">
                <li>Scikit-learn</li>
                <li>Logistic Regression</li>
                <li>StandardScaler</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with tech3:
        st.markdown(f"""
        <div class="card">
            <h4>Visualization</h4>
            <ul style="font-size: 0.95rem;">
                <li>Matplotlib</li>
                <li>Seaborn</li>
                <li>Custom CSS</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # ==== DATASET INFO ====
    st.markdown(f"""
    <div class="card">
        <h3>📊 Dataset Information</h3>
        <p><b>Training Data:</b> 70,000+ patient records with 12 features</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="card">
            <h5>📍 Patient Features:</h5>
            <ul style="font-size: 0.9rem;">
                <li>Age, Gender, Height, Weight</li>
                <li>Systolic & Diastolic BP</li>
                <li>Cholesterol & Glucose Levels</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="card">
            <h5>🏃 Lifestyle Factors:</h5>
            <ul style="font-size: 0.9rem;">
                <li>Smoking Status</li>
                <li>Alcohol Consumption</li>
                <li>Physical Activity Level</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # ==== PROJECT PURPOSE ====
    st.markdown(f"""
    <div class="card">
        <h3>🎓 Educational Purpose</h3>
        <p style="font-size: 1rem; line-height: 1.7;">
        This project was developed for the <b>Machine Learning & Deep Learning (MLDL)</b> curriculum 
        to demonstrate practical applications of ML in healthcare analytics. It showcases data preprocessing, 
        model training, evaluation, and deployment using modern web technologies.
        </p>
        <br>
        <p style="font-size: 0.95rem; color: #dc2626; font-weight: bold;">
        ⚠️ This is an educational tool only - NOT for medical diagnosis or clinical use.
        </p>
    </div>
    """, unsafe_allow_html=True)



# #------Disclimer-----
# elif st.session_state['page'] == 'Disclaimer':

#     st.markdown(
#         """
#         <h1 style='text-align:center; color:#b91c1c;'>⚠️ Medical & Ethical Disclaimer</h1>
#         """,
#         unsafe_allow_html=True
#     )

#     st.markdown("---")

#     st.markdown(
#         """
#         ## 📌 Purpose of This Application
#         This application is developed **strictly for academic, learning, and demonstration purposes** 
#         as part of a **Machine Learning & Deep Learning (MLDL) project**.
#         It showcases how machine learning models can be applied to healthcare-related datasets.

#         The system **does NOT provide medical diagnosis, treatment, or prevention advice**.

#         ---
#         ## 🧠 Model & Prediction Limitations
#         - The predictions are generated using a **statistical machine learning model**
#         - The model is trained on **historical and limited datasets**
#         - Predictions may be affected by:
#           - Incomplete or incorrect user input
#           - Data bias present in the training dataset
#           - Model assumptions and simplifications
#         - The model **does not guarantee accuracy or reliability**

#         ---
#         ## 🩺 No Medical Advice
#         This application **must not be used** as a substitute for:
#         - Professional medical advice
#         - Clinical diagnosis
#         - Treatment planning
#         - Emergency medical services

#         Always consult a **licensed physician or qualified healthcare provider**
#         regarding any medical condition.

#         ---
#         ## ⚠️ Risk & Responsibility
#         - Users are fully responsible for how they interpret and use the results
#         - The developer, institution, and contributors **bear no liability**
#         for decisions made based on this application
#         - Any action taken based on the prediction is **solely at the user’s risk**

#         ---
#         ## 📊 Data Usage & Privacy
#         - User input data is **not permanently stored**
#         - No personal data is shared with third parties
#         - This application does **not perform real-time patient monitoring**

#         ---
#         ## ⚖️ Legal Notice
#         This software is provided **“as is” without warranty of any kind**.
#         There is **no expressed or implied guarantee** regarding:
#         - Accuracy
#         - Completeness
#         - Suitability for real-world medical use

#         ---
#         ## 🎓 Academic Declaration
#         This project is created as part of an **educational curriculum**
#         to demonstrate:
#         - Data preprocessing
#         - Feature engineering
#         - Model training & evaluation
#         - ML deployment using Streamlit

#         The application **must not be deployed in clinical or commercial environments**.
#         """
#     )

#     st.markdown("---")

#     st.warning(
#         "⚠️ By continuing to use this application, you confirm that you understand "
#         "this is an educational tool and not a medical diagnostic system."
#     )



# ------ Disclaimer -----
elif st.session_state['page'] == 'Disclaimer':

    st.markdown(
        "<h1 style='text-align:center; color:#b91c1c;'>⚠️ Medical & Ethical Disclaimer</h1>",
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer-card">
        <h2>📌 Purpose of This Application</h2>
        <p>
            This application is developed <strong>strictly for academic, learning, and demonstration purposes</strong>
            as part of a <strong>Machine Learning & Deep Learning (MLDL) project</strong>.
            It demonstrates how machine learning models can be applied to healthcare-related datasets.
        </p>
        <p>
            This system <strong>does NOT provide medical diagnosis, treatment, or prevention advice</strong>.
        </p>
    </div>

    <div class="disclaimer-card">
        <h2>🧠 Model & Prediction Limitations</h2>
        <ul>
            <li>Predictions are generated using a statistical machine learning model</li>
            <li>The model is trained on historical and limited datasets</li>
            <li>Results may be affected by incomplete input or dataset bias</li>
            <li>The model does <strong>not guarantee accuracy or reliability</strong></li>
        </ul>
    </div>

    <div class="disclaimer-card">
        <h2>🩺 No Medical Advice</h2>
        <p>
            This application <strong>must not be used</strong> as a substitute for professional medical advice,
            diagnosis, treatment, or emergency healthcare services.
        </p>
        <p>
            Always consult a <strong>licensed physician or qualified healthcare provider</strong>.
        </p>
    </div>

    <div class="disclaimer-card">
        <h2>⚠️ Risk & Responsibility</h2>
        <ul>
            <li>Users are fully responsible for interpreting predictions</li>
            <li>Developers and institutions bear <strong>no liability</strong></li>
            <li>All actions taken are at the user’s own risk</li>
        </ul>
    </div>

    <div class="disclaimer-card">
        <h2>📊 Data Usage & Privacy</h2>
        <ul>
            <li>User input is not permanently stored</li>
            <li>No personal data is shared with third parties</li>
            <li>No real-time patient monitoring is performed</li>
        </ul>
    </div>

    <div class="disclaimer-card">
        <h2>⚖️ Legal Notice</h2>
        <p>
            This software is provided <strong>“as is” without warranty of any kind</strong>.
            There is no expressed or implied guarantee of accuracy or suitability
            for real-world medical use.
        </p>
    </div>

    <div class="disclaimer-card">
        <h2>🎓 Academic Declaration</h2>
        <p>
            This project is part of an educational curriculum demonstrating data preprocessing,
            feature engineering, model training, evaluation, and deployment using Streamlit.
        </p>
        <p>
            The application <strong>must not be deployed in clinical or commercial environments</strong>.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.warning(
        "⚠️ By continuing to use this application, you confirm that this is an educational tool "
        "and not a medical diagnostic system."
    )

    