import streamlit as st
import base64
import os
from inference import predict_salary

def get_base64_img(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return ""

bg_image_base64 = get_base64_img("plots/hero.png")

# Page Configuration
st.set_page_config(
    page_title="Salary Predictor",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom 3D CSS for a Glassmorphism UI & Massive Typography
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@900&display=swap');
    
    /* Dark Premium Background gradient */
    .stApp {
        background: #111111 !important;
        color: #ffffff;
    }
    
    /* Massive Typography Hero Section */
    .hero-container {
        position: relative;
        width: 100%;
        height: 85vh;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        overflow: visible;
        margin-top: -80px;
        margin-bottom: 20px;
        background-color: transparent !important;
        background-image: url("data:image/png;base64,BG_PLACEHOLDER");
        background-size: 42% auto;
        background-position: 15% center;
        background-repeat: no-repeat;
        border-radius: 0px;
        box-shadow: none;
    }
    
    /* Remove overlay so the image is fully vibrant */
    .hero-container::before {
        display: none;
    }
    
    .hero-main-text {
        font-family: 'Montserrat', sans-serif;
        font-size: 14vw !important;
        font-weight: 900 !important;
        color: #ffffff;
        line-height: 0.8 !important;
        text-transform: uppercase;
        letter-spacing: -3px;
        z-index: 1;
        text-align: center;
        animation: slideUp 1.2s cubic-bezier(0.16, 1, 0.3, 1) forwards;
    }
    
    .hero-sub-text {
        font-family: 'Brush Script MT', 'cursive', sans-serif;
        font-size: 10vw !important;
        color: #9b59b6;
        text-transform: none;
        position: relative;
        margin-top: -5vw;
        margin-left: 25vw;
        z-index: 2;
        transform: rotate(-3deg);
        text-shadow: 4px 4px 10px rgba(0,0,0,0.7);
        animation: fadeInScale 1.5s ease-out forwards;
        animation-delay: 0.5s;
        opacity: 0;
    }
    
    .hero-desc-text {
        position: absolute;
        bottom: 8%;
        right: 15%;
        width: 25vw;
        color: #e0e0e0;
        font-family: 'Arial', sans-serif;
        font-size: 1.1vw !important;
        line-height: 1.6;
        z-index: 2;
        text-align: left;
        animation: slideUp 1.5s ease-out forwards;
        animation-delay: 0.8s;
        opacity: 0;
    }

    .hero-star {
        position: absolute;
        bottom: 3%;
        right: 5%;
        font-size: 5vw;
        color: white;
        z-index: 2;
        animation: spin 10s linear infinite;
    }
    
    @keyframes spin { 100% { transform: rotate(360deg); } }
    @keyframes slideUp {
        0% { transform: translateY(150px); opacity: 0; }
        100% { transform: translateY(0); opacity: 1; }
    }
    @keyframes fadeInScale {
        0% { transform: scale(0.8); opacity: 0; }
        100% { transform: scale(1); opacity: 1; }
    }
    
    /* Section Divider */
    .section-label {
        font-family: 'Montserrat', sans-serif;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 4px;
        text-transform: uppercase;
        color: #555;
        margin-bottom: 60px;
        border-top: 1px solid #222;
        padding-top: 30px;
    }

    .section-title {
        font-family: 'Montserrat', sans-serif;
        font-size: 3.5vw !important;
        font-weight: 900;
        color: #ffffff !important;
        line-height: 1.1;
        margin-bottom: 10px;
    }

    /* Clean global label & text */
    label, p, h1, h2, h3 {
        color: #ffffff !important;
    }
    label {
        font-family: 'Montserrat', sans-serif !important;
        font-size: 11px !important;
        font-weight: 700 !important;
        letter-spacing: 2px !important;
        text-transform: uppercase !important;
        color: #666 !important;
    }

    /* Minimalist inputs - underline only */
    div[data-testid="stNumberInput"] input {
        background: transparent !important;
        border: none !important;
        border-bottom: 1px solid rgba(255,255,255,0.15) !important;
        border-radius: 0 !important;
        color: #ffffff !important;
        font-family: 'Montserrat', sans-serif !important;
        font-size: 22px !important;
        font-weight: 600 !important;
        padding: 8px 0 !important;
        box-shadow: none !important;
    }
    div[data-testid="stNumberInput"] input:focus {
        border-bottom-color: #9b59b6 !important;
        outline: none !important;
        box-shadow: none !important;
    }
    div[data-testid="stSelectbox"] > div > div {
        background: transparent !important;
        border: none !important;
        border-bottom: 1px solid rgba(255,255,255,0.15) !important;
        border-radius: 0 !important;
        color: #ffffff !important;
        font-family: 'Montserrat', sans-serif !important;
        font-size: 18px !important;
        box-shadow: none !important;
    }

    /* Form container */
    div[data-testid="stForm"] {
        background: rgba(255,255,255,0.015);
        border-top: 1px solid rgba(255,255,255,0.07);
        border-bottom: 1px solid rgba(255,255,255,0.07);
        border-left: none;
        border-right: none;
        border-radius: 0;
        padding: 50px 20px;
    }

    /* Tabs */
    div[data-testid="stTabs"] button {
        font-family: 'Montserrat', sans-serif !important;
        font-size: 12px !important;
        font-weight: 700 !important;
        letter-spacing: 2px !important;
        text-transform: uppercase !important;
        color: #444 !important;
        padding: 12px 24px !important;
    }
    div[data-testid="stTabs"] button[aria-selected="true"] {
        color: #ffffff !important;
        border-bottom: 2px solid #ffffff !important;
    }
    div[data-testid="stTabs"] button:hover {
        color: #aaa !important;
        background: transparent !important;
    }

    /* Submit Button */
    .stButton > button {
        width: 100%;
        background: #ffffff !important;
        color: #000000 !important;
        font-family: 'Montserrat', sans-serif !important;
        font-size: 13px !important;
        font-weight: 800 !important;
        text-transform: uppercase;
        letter-spacing: 3px;
        border: none !important;
        border-radius: 0 !important;
        padding: 18px 0 !important;
        margin-top: 30px;
        transition: background 0.3s ease, color 0.3s ease;
    }
    .stButton > button:hover {
        background: #9b59b6 !important;
        color: #ffffff !important;
    }

    /* Result card */
    .result-card {
        border-top: 1px solid rgba(255,255,255,0.08);
        padding: 50px 20px;
        margin-top: 40px;
        text-align: left;
    }
    .result-label {
        font-family: 'Montserrat', sans-serif;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 4px;
        text-transform: uppercase;
        color: #555;
        margin-bottom: 12px;
    }
    .result-text {
        font-family: 'Montserrat', sans-serif;
        font-size: 5vw !important;
        font-weight: 900;
        color: #ffffff;
        line-height: 1;
        letter-spacing: -2px;
    }
    </style>
""".replace("BG_PLACEHOLDER", bg_image_base64), unsafe_allow_html=True)

st.markdown("""
<div class="hero-container">
    <div class="hero-main-text">SALARY</div>
    <div class="hero-sub-text">Predictor</div>
    <div class="hero-desc-text">
        This is an advanced AI-powered forecasting engine.<br><br>
        Once details are expertly provided, our mathematical algorithms predict precise market value and earning potential in the Indian sector.
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<p class="section-label">Employee Intelligence Engine</p>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Salary Predictor", "Data Insights"])

with tab1:
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            age_input = st.number_input("Age", min_value=18, max_value=70, value=25, step=1)
            education_input = st.selectbox("Education Level", ["High School", "Bachelor", "Master", "PhD"], index=1)
            city_input = st.selectbox("City Tier", ["Tier 1", "Tier 2", "Tier 3"], index=0)
            
        with col2:
            exp_input = st.number_input("Years of Experience", min_value=0, max_value=50, value=3, step=1)
            role_input = st.selectbox("Job Role", ["Junior", "Mid", "Senior", "Lead", "Manager"], index=1)

        submitted = st.form_submit_button("Predict Salary")

    if submitted:
        # Ensure Years of Experience logically aligns with Age (cannot have 10 years exp at 18)
        if exp_input > (age_input - 18):
            st.error(f"Error: Years of experience ({exp_input}) is too high for an age of {age_input}.")
        else:
            sample_data = {
                "Age": age_input,
                "Years_of_Experience": exp_input,
                "Education_Level": education_input,
                "Job_Role": role_input,
                "City_Tier": city_input,
            }
            
            try:
                with st.spinner("Analyzing market data..."):
                    salary = predict_salary(sample_data)
                    
                st.markdown(f"""
                    <div class="result-card">
                        <div class="result-label">Estimated Annual Package</div>
                        <div class="result-text">&#8377;{salary:,.0f}</div>
                    </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"An error occurred: {e}. Please ensure the model is trained by running main.py first.")

with tab2:
    st.markdown('<p class="section-label">Data Insights & Report</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-title">How the data<br>shapes salary</p>', unsafe_allow_html=True)
    
    # Add Download Report Button
    if os.path.exists("Project_Report.html"):
        with open("Project_Report.html", "r", encoding="utf-8") as f:
            report_data = f.read()
        st.download_button(
            label="DOWNLOAD PROJECT REPORT (.HTML)",
            data=report_data,
            file_name="Project_Report.html",
            mime="text/html"
        )
        
    st.markdown("<p style='font-size:15px; color:#555; margin-bottom:40px; margin-top:20px;'>Charts generated from the training pipeline — showing distribution across education, roles and city tiers.</p>", unsafe_allow_html=True)
    
    st.image("plots/salary_by_categories.png", use_container_width=True, caption="Salary distribution across Education, Job Roles, and City Tiers.")
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.image("plots/salary_distribution.png", use_container_width=True, caption="Overall Annual Salary Spread")
    with col_b:
        st.image("plots/correlation_matrix.png", use_container_width=True, caption="Feature Correlation Matrix")
