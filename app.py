import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="CardioPredict AI",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# LOAD TRAINED MODEL & SCALER
# =========================

model = joblib.load("lr_model.pkl")  
scaler = joblib.load("scaler.pkl")

# =========================
# CUSTOM CSS - MEDICAL THEME
# =========================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #f0f4f8 0%, #e8eef3 100%);
    }
    
    h1 {
        color: #1e3a5f;
        font-weight: 700;
        font-size: 2.8rem !important;
        margin-bottom: 0.5rem;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    h2 {
        color: #2c5282;
        font-weight: 600;
        margin-top: 2rem;
    }
    
    h3 {
        color: #3d5a80;
        font-weight: 500;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f7fafc 100%);
        padding: 25px;
        border-radius: 12px;
        border: 2px solid #e2e8f0;
        box-shadow: 0 4px 15px rgba(30, 58, 95, 0.08);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        margin-bottom: 20px;
    }
    
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(30, 58, 95, 0.12);
        border-color: #0891b2;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #0891b2;
        margin: 10px 0;
    }
    
    .metric-label {
        color: #64748b;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        font-weight: 600;
    }
    
    .prediction-card {
        background: #ffffff;
        padding: 35px;
        border-radius: 16px;
        border: 2px solid #e2e8f0;
        box-shadow: 0 6px 20px rgba(30, 58, 95, 0.1);
        margin: 20px 0;
    }
    
    .risk-high {
        background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%);
        padding: 25px;
        border-radius: 12px;
        color: white;
        text-align: center;
        font-size: 1.5rem;
        font-weight: 600;
        box-shadow: 0 6px 20px rgba(220, 38, 38, 0.3);
        border: 3px solid #991b1b;
    }
    
    .risk-low {
        background: linear-gradient(135deg, #059669 0%, #047857 100%);
        padding: 25px;
        border-radius: 12px;
        color: white;
        text-align: center;
        font-size: 1.5rem;
        font-weight: 600;
        box-shadow: 0 6px 20px rgba(5, 150, 105, 0.3);
        border: 3px solid #065f46;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #0891b2 0%, #0e7490 100%);
        color: white;
        border: none;
        padding: 14px 40px;
        border-radius: 8px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(8, 145, 178, 0.3);
        width: 100%;
        border: 2px solid #0e7490;
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #0e7490 0%, #155e75 100%);
        transform: translateY(-2px);
        box-shadow: 0 6px 18px rgba(8, 145, 178, 0.4);
    }
    
    .info-box {
        background: #f0f9ff;
        border-left: 5px solid #0891b2;
        padding: 18px;
        border-radius: 8px;
        margin: 15px 0;
        color: #1e3a5f;
        box-shadow: 0 2px 8px rgba(8, 145, 178, 0.1);
    }
    
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e3a5f 0%, #0f172a 100%);
    }
    
    div[data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }
    
    .stSelectbox, .stNumberInput {
        background: white;
    }
    
    label {
        color: #334155 !important;
        font-weight: 500 !important;
    }
</style>
""", unsafe_allow_html=True)

# =========================
# LOAD DATA
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

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.markdown("""
    <div style='text-align: center; padding: 20px 0; border-bottom: 2px solid rgba(255,255,255,0.1);'>
        <h2 style='color: #ffffff !important; margin: 0; font-size: 1.8rem;'>🫀 CardioPredict</h2>
        <p style='color: #94a3b8 !important; font-size: 0.9rem; margin: 5px 0 0 0;'>Medical AI Assistant</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    menu = st.radio("", ["🏠 Dashboard", "🔮 Prediction", "📊 Analytics", "⚙️ Model Info"], label_visibility="collapsed")
    st.markdown("---")
    
    if not df.empty:
        with st.expander("📋 Data Information", expanded=False):
            st.markdown(f"""
            <div style='font-size: 0.85rem;'>
                <strong>Total Records:</strong> {len(df):,}<br>
                <strong>Features:</strong> {len(df.columns)}<br>
                <strong>Target Variable:</strong> cardio<br>
                <strong>Model Accuracy:</strong> 72.24%
            </div>
            """, unsafe_allow_html=True)

# =========================
# DASHBOARD
# =========================
if menu == "🏠 Dashboard":
    st.markdown("<h1>🫀 Cardiovascular Health Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #64748b; font-size: 1.1rem;'>Comprehensive overview of patient data and health metrics</p>", unsafe_allow_html=True)
    
    if not df.empty:
        total_patients = len(df)
        positive = int(df["cardio"].sum())
        negative = total_patients - positive
        risk_rate = (positive / total_patients) * 100
        MODEL_ACCURACY = 72.24 #%
        
        # Top metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Total Patients</div>
                <div class='metric-value'>{total_patients:,}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>High Risk</div>
                <div class='metric-value' style='background: linear-gradient(120deg, #ff6b6b, #ee5a6f); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>{positive:,}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Low Risk</div>
                <div class='metric-value' style='background: linear-gradient(120deg, #51cf66, #37b24d); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>{negative:,}</div>
            </div>
            """, unsafe_allow_html=True)
        
        

        with col4:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Model Accuracy</div>
                <div class='metric-value'>{MODEL_ACCURACY:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)

        
        st.markdown("<br>", unsafe_allow_html=True)
        
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure(go.Pie(labels=['Positive', 'Negative'], values=[positive, negative], hole=0.6,
                                   marker=dict(colors=['#dc2626', '#059669'])))
            fig.update_layout(title="Disease Distribution", paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='#334155'), height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with c2:
            st.markdown(f"<h3>📊 Dataset Statistics</h3>", unsafe_allow_html=True)
            stats = df[['age_in_year', 'height', 'weight', 'ap_hi', 'ap_lo', 'bmi']].describe()
            st.dataframe(stats.T[['mean', 'std', 'min', 'max']].round(2), use_container_width=True)
        
# =========================
# ANALYTICS
# =========================
elif menu == "📊 Analytics":
    st.markdown("<h1>📊 Advanced Analytics</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #64748b; font-size: 1.1rem;'>Deep dive into cardiovascular risk factors and patterns</p>", unsafe_allow_html=True)
    
    if not df.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            # Update all analytics chart colors to medical theme
            chol_risk = df.groupby(['cholesterol', 'cardio']).size().unstack(fill_value=0)
            fig_chol = go.Figure()
            fig_chol.add_trace(go.Bar(
                x=['Normal', 'Above Normal', 'High'],
                y=chol_risk[0] if 0 in chol_risk.columns else [0, 0, 0],
                name='Low Risk',
                marker_color='#059669'
            ))
            fig_chol.add_trace(go.Bar(
                x=['Normal', 'Above Normal', 'High'],
                y=chol_risk[1] if 1 in chol_risk.columns else [0, 0, 0],
                name='High Risk',
                marker_color='#dc2626'
            ))
            fig_chol.update_layout(
                title=dict(text="Cholesterol Levels vs Risk", font=dict(size=18, color='#1e3a5f')),
                xaxis_title="Cholesterol Level",
                yaxis_title="Count",
                barmode='group',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#334155'),
                height=400
            )
            st.plotly_chart(fig_chol, use_container_width=True)
        
        with col2:
            # Blood Pressure Analysis
            fig_bp = go.Figure()
            for cardio_status, color, name in [(0, '#059669', 'Low Risk'), (1, '#dc2626', 'High Risk')]:
                bp_data = df[df['cardio'] == cardio_status].sample(min(1000, len(df[df['cardio'] == cardio_status])))
                fig_bp.add_trace(go.Scatter(
                    x=bp_data['ap_hi'],
                    y=bp_data['ap_lo'],
                    mode='markers',
                    name=name,
                    marker=dict(color=color, size=5, opacity=0.6)
                ))
            fig_bp.update_layout(
                title=dict(text="Blood Pressure Pattern Analysis", font=dict(size=18, color='#1e3a5f')),
                xaxis_title="Systolic BP",
                yaxis_title="Diastolic BP",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#334155'),
                height=400
            )
            st.plotly_chart(fig_bp, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Lifestyle factors
            lifestyle_data = {
                'Factor': ['Smoking', 'Alcohol', 'Physical Activity'],
                'High Risk': [
                    df[df['cardio'] == 1]['smoke'].sum(),
                    df[df['cardio'] == 1]['alco'].sum(),
                    df[df['cardio'] == 1]['active'].sum()
                ],
                'Low Risk': [
                    df[df['cardio'] == 0]['smoke'].sum(),
                    df[df['cardio'] == 0]['alco'].sum(),
                    df[df['cardio'] == 0]['active'].sum()
                ]
            }
            
            fig_lifestyle = go.Figure()
            fig_lifestyle.add_trace(go.Bar(
                x=lifestyle_data['Factor'],
                y=lifestyle_data['Low Risk'],
                name='Low Risk',
                marker_color='#059669'
            ))
            fig_lifestyle.add_trace(go.Bar(
                x=lifestyle_data['Factor'],
                y=lifestyle_data['High Risk'],
                name='High Risk',
                marker_color='#dc2626'
            ))
            fig_lifestyle.update_layout(
                title=dict(text="Lifestyle Factors Impact", font=dict(size=18, color='#1e3a5f')),
                xaxis_title="Factor",
                yaxis_title="Count",
                barmode='group',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#334155'),
                height=400
            )
            st.plotly_chart(fig_lifestyle, use_container_width=True)
        
        with col2:
            # Glucose vs Risk
            gluc_risk = df.groupby(['gluc', 'cardio']).size().unstack(fill_value=0)
            fig_gluc = go.Figure()
            fig_gluc.add_trace(go.Bar(
                x=['Normal', 'Above Normal', 'High'],
                y=gluc_risk[0] if 0 in gluc_risk.columns else [0, 0, 0],
                name='Low Risk',
                marker_color='#059669'
            ))
            fig_gluc.add_trace(go.Bar(
                x=['Normal', 'Above Normal', 'High'],
                y=gluc_risk[1] if 1 in gluc_risk.columns else [0, 0, 0],
                name='High Risk',
                marker_color='#dc2626'
            ))
            fig_gluc.update_layout(
                title=dict(text="Glucose Levels vs Risk", font=dict(size=18, color='#1e3a5f')),
                xaxis_title="Glucose Level",
                yaxis_title="Count",
                barmode='group',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#334155'),
                height=400
            )
            st.plotly_chart(fig_gluc, use_container_width=True)
        
        # Correlation heatmap
        st.markdown("### 🔥 Feature Correlation Matrix")
        numeric_cols = ['age_in_year', 'height', 'weight', 'ap_hi', 'ap_lo', 'bmi', 'cholesterol', 'gluc']
        corr_matrix = df[numeric_cols].corr()
        
        fig_corr = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='Teal',
            zmid=0,
            text=corr_matrix.values.round(2),
            texttemplate='%{text}',
            textfont={"size": 10, "color": "white"},
            colorbar=dict(title="Correlation", titlefont=dict(color='#1e3a5f'))
        ))
        fig_corr.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#334155'),
            height=500
        )
        st.plotly_chart(fig_corr, use_container_width=True)

# =========================
# MODEL INFO
# =========================
elif menu == "⚙️ Model Info":
    st.markdown("<h1>⚙️ Model Information</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #64748b; font-size: 1.1rem;'>Technical specifications and performance metrics</p>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class='metric-card'>
            <div class='metric-label'>Model Type</div>
            <div class='metric-value' style='font-size: 1.2rem;'>Logistic Regression</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='metric-card'>
            <div class='metric-label'>Test Accuracy</div>
            <div class='metric-value'>72.24%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='metric-card'>
            <div class='metric-label'>Features</div>
            <div class='metric-value' style='font-size: 1.5rem;'>12</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class='metric-card'>
            <div class='metric-label'>Training Samples</div>
            <div class='metric-value' style='font-size: 1.5rem;'>54,537</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Model performance metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 Performance Metrics")
        metrics_data = {
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC'],
            'Score': [0.7224, 0.7150, 0.7580, 0.7359, 0.7950]
        }
        
        fig_metrics = go.Figure()
        fig_metrics.add_trace(go.Bar(
            x=metrics_data['Metric'],
            y=metrics_data['Score'],
            marker_color='#0891b2',
            text=[f"{score:.1%}" for score in metrics_data['Score']],
            textposition='outside'
        ))
        fig_metrics.update_layout(
            yaxis_title="Score",
            yaxis_range=[0, 1],
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#334155'),
            height=400
        )
        st.plotly_chart(fig_metrics, use_container_width=True)
    
    with col2:
        st.markdown("### 🎯 Feature Importance")
        
        feature_names = ['Gender', 'Height', 'Weight', 'Systolic BP', 'Diastolic BP', 
                        'Cholesterol', 'Glucose', 'Smoking', 'Alcohol', 'Activity', 'Age', 'BMI']
        importance = np.abs(w.flatten())
        feature_importance = sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True)
        
        fig_importance = go.Figure()
        fig_importance.add_trace(go.Bar(
            y=[f[0] for f in feature_importance],
            x=[f[1] for f in feature_importance],
            orientation='h',
            marker_color='#0e7490',
            text=[f"{f[1]:.3f}" for f in feature_importance],
            textposition='outside'
        ))
        fig_importance.update_layout(
            xaxis_title="Importance Score",
            yaxis_title="",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#334155'),
            height=400
        )
        st.plotly_chart(fig_importance, use_container_width=True)
    
    # Training details
    st.markdown("### 📝 Training Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class='info-box'>
            <strong>🔧 Algorithm:</strong> Logistic Regression with Gradient Descent<br><br>
            <strong>📊 Training Data:</strong> 54,537 samples<br><br>
            <strong>✅ Test Data:</strong> 13,635 samples<br><br>
            <strong>⚡ Optimization:</strong> Gradient Descent
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='info-box'>
            <strong>🎯 Learning Rate:</strong> 0.1<br><br>
            <strong>🔄 Iterations:</strong> 20,000<br><br>
            <strong>📐 Preprocessing:</strong> Standard Scaling (Z-score)<br><br>
            <strong>🎲 Random State:</strong> 42
        </div>
        """, unsafe_allow_html=True)

# =========================
# PREDICTION UI
# =========================
elif menu == "🔮 Prediction":
    st.markdown("<h1>🔮 Cardio Risk Prediction</h1>", unsafe_allow_html=True)
    
    # # Prominent disclaimer at the top
    # st.markdown("""
    # <div style='background: linear-gradient(135deg, #fef2f2, #fee2e2); 
    #             border-left: 5px solid #dc2626; padding: 20px; border-radius: 10px; margin-bottom: 25px; border: 2px solid #fca5a5;'>
    #     <h3 style='color: #991b1b; margin: 0 0 10px 0; font-size: 1.2rem;'>⚠️ Medical Disclaimer</h3>
    #     <p style='color: #7f1d1d; font-size: 0.9rem; margin: 0; line-height: 1.6;'>
    #         <strong>This is a research and educational tool only.</strong> It is NOT intended for medical diagnosis or treatment decisions. 
    #         The model has 72.24% accuracy and predictions may be incorrect. Always consult qualified healthcare professionals 
    #         for medical advice. Do not make health decisions based solely on this tool.
    #     </p>
    # </div>
    # """, unsafe_allow_html=True)
    
    with st.form("predict_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 👤 Demographics")
            age_years = st.number_input("Age (Years)", 18, 100, 50, help="Patient's age in years")
            gender = st.selectbox("Gender", [1, 2], format_func=lambda x: "Female" if x == 1 else "Male")
            height = st.number_input("Height (cm)", 140, 210, 165, help="Height in centimeters")
            weight = st.number_input("Weight (kg)", 40, 200, 70, help="Weight in kilograms")
        
        with col2:
            st.markdown("#### 🩺 Clinical Measurements")
            ap_hi = st.number_input("Systolic BP (mmHg)", 90, 250, 120, help="Upper blood pressure reading")
            ap_lo = st.number_input("Diastolic BP (mmHg)", 60, 150, 80, help="Lower blood pressure reading")
            cholesterol = st.selectbox("Cholesterol Level", [1, 2, 3], 
                                      format_func=lambda x: ["Normal", "Above Normal", "High"][x-1])
            gluc = st.selectbox("Glucose Level", [1, 2, 3],
                               format_func=lambda x: ["Normal", "Above Normal", "High"][x-1])
        
        with col3:
            st.markdown("#### 🏃 Lifestyle Factors")
            smoke = st.selectbox("Smoking Status", [0, 1], 
                                format_func=lambda x: "Non-Smoker" if x == 0 else "Smoker")
            alco = st.selectbox("Alcohol Consumption", [0, 1],
                               format_func=lambda x: "No" if x == 0 else "Yes")
            active = st.selectbox("Physical Activity", [0, 1],
                                 format_func=lambda x: "Inactive" if x == 0 else "Active")
            
            # Calculate BMI
            bmi = weight / ((height / 100) ** 2)
            
            # st.markdown(f"""
            # <div class='info-box' style='margin-top: 20px;'>
            #     <strong>Calculated BMI:</strong> {bmi:.1f}<br>
            #     <strong>Category:</strong> {bmi_category}
            # </div>
            # """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            submit = st.form_submit_button("🔍 Analyze Risk")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    if submit:
        # Prepare input data in the exact order as training
        # Features: gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active, age_in_year, bmi
        user_data = np.array([[gender, height, weight, ap_hi, ap_lo, cholesterol, 
                              gluc, smoke, alco, active, age_years, bmi]])
        
        # Scale the data
        user_data_scaled = scaler.transform(user_data)
        
        # Predict
        # prediction= model.predict(user_data_scaled)
        prediction = model.predict(user_data_scaled)[0]
        probability = model.predict_proba(user_data_scaled)[0][1]
        
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
        
        # with col2:
        #     # Risk gauge chart
        #     fig_gauge = go.Figure(go.Indicator(
        #         mode="gauge+number+delta",
        #         value=probability * 100,
        #         domain={'x': [0, 1], 'y': [0, 1]},
        #         title={'text': "Risk Score", 'font': {'color': '#1e3a5f', 'size': 20}},
        #         delta={'reference': 50, 'increasing': {'color': "#dc2626"}},
        #         gauge={
        #             'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#334155"},
        #             'bar': {'color': "#dc2626" if prediction == 1 else "#059669"},
        #             'bgcolor': "white",
        #             'borderwidth': 2,
        #             'bordercolor': "#cbd5e1",
        #             'steps': [
        #                 {'range': [0, 50], 'color': '#d1fae5'},
        #                 {'range': [50, 100], 'color': '#fee2e2'}
        #             ],
        #             'threshold': {
        #                 'line': {'color': "#64748b", 'width': 4},
        #                 'thickness': 0.75,
        #                 'value': 50
        #             }
        #         }
        #     ))
            
        #     fig_gauge.update_layout(
        #         paper_bgcolor='rgba(0,0,0,0)',
        #         plot_bgcolor='rgba(0,0,0,0)',
        #         font={'color': "#334155", 'family': "Poppins"},
        #         height=300,
        #         margin=dict(l=20, r=20, t=40, b=20)
        #     )
        #     st.plotly_chart(fig_gauge, use_container_width=True)
            
            # # Patient summary
            # st.markdown(f"""
            # <div class='info-box'>
            #     <strong>📋 Patient Summary</strong><br><br>
            #     <strong>Age:</strong> {age_years} years<br>
            #     <strong>Gender:</strong> {'Female' if gender == 1 else 'Male'}<br>
            #     <strong>BMI:</strong> {bmi:.1f}<br>
            #     <strong>BP:</strong> {ap_hi}/{ap_lo} mmHg<br>
            #     <strong>Lifestyle:</strong> {'Smoker' if smoke else 'Non-smoker'}, 
            #     {'Active' if active else 'Inactive'}
            # </div>
            # """, unsafe_allow_html=True)
        
        # Feature contribution chart
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 📊 Risk Factor Contribution Analysis")
        
        feature_names = ['Gender', 'Height', 'Weight', 'Systolic BP', 'Diastolic BP', 
                        'Cholesterol', 'Glucose', 'Smoking', 'Alcohol', 'Activity', 'Age', 'BMI']
        w = model.coef_[0]
        contributions = user_data_scaled[0] * w.flatten()
        
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
        
        st.markdown("""
        <div class='info-box'>
            <strong>ℹ️ Understanding This Chart:</strong><br>
            This chart shows how each factor contributes to the overall risk prediction. 
            Red bars increase risk (push toward disease), while green bars decrease risk (push toward health). 
            Longer bars have stronger influence on the prediction.<br><br>
            <em style='color: #ff6b6b; font-size: 0.85rem;'>⚠️ Disclaimer: This analysis is for educational purposes only and should not replace professional medical assessment.</em>
        </div>
        """, unsafe_allow_html=True)
        
        # Final disclaimer at bottom
        st.markdown("""
        <div style='background: #fef2f2; border: 2px solid #dc2626; 
                    padding: 20px; border-radius: 12px; margin-top: 30px; text-align: center;'>
            <h4 style='color: #991b1b; margin: 0 0 15px 0;'>⚠️ CRITICAL REMINDER</h4>
            <p style='color: #7f1d1d; font-size: 0.95rem; margin: 0; line-height: 1.8;'>
                <strong>This AI tool is NOT a medical device and should NEVER be used for self-diagnosis.</strong><br>
                If you have any health concerns or symptoms, please consult a qualified healthcare provider immediately.<br>
                In case of emergency, call your local emergency services.<br><br>
                <span style='font-size: 0.85rem; color: #991b1b;'>
                    Model Accuracy: 72.24% | For Research & Educational Use Only | Not FDA Approved
                </span>
            </p>
        </div>
        """, unsafe_allow_html=True)











