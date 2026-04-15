import streamlit as st
import base64
import os
import sys
import sklearn
from inference import predict_salary

st.sidebar.write("Python Path:", sys.executable)
st.sidebar.write("Sklearn Version:", sklearn.__version__)

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

# Custom CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@900&display=swap');
    
    .stApp {
        background: #111111 !important;
        color: #ffffff;
    }
    
    .hero-container {
        position: relative;
        width: 100%;
        height: 85vh;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        margin-top: -80px;
        margin-bottom: 20px;
        background-image: url("data:image/png;base64,BG_PLACEHOLDER");
        background-size: 42% auto;
        background-position: 15% center;
        background-repeat: no-repeat;
    }
    
    .hero-main-text {
        font-family: 'Montserrat', sans-serif;
        font-size: 14vw !important;
        font-weight: 900 !important;
        color: #ffffff;
        line-height: 0.8 !important;
        text-transform: uppercase;
        letter-spacing: -3px;
        text-align: center;
    }
    
    .hero-sub-text {
        font-family: 'Brush Script MT', 'cursive', sans-serif;
        font-size: 10vw !important;
        color: #9b59b6;
        margin-top: -5vw;
        margin-left: 25vw;
        transform: rotate(-3deg);
    }
    
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
    }

    label, p, h1, h2, h3 {
        color: #ffffff !important;
    }

    /* INPUT FIELDS (Age, Experience): BLACK BG + WHITE TEXT */
    div[data-testid="stNumberInput"] div[data-baseweb="input"],
    div[data-testid="stNumberInput"] input {
        background-color: #000000 !important;
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        border-bottom: 1px solid rgba(255,255,255,0.3) !important;
        border-radius: 0 !important;
    }

    /* DROPDOWNS (Education, Role, City): BLACK BG + WHITE TEXT */
    div[data-testid="stSelectbox"] div[data-baseweb="select"],
    div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
        background-color: #000000 !important;
        border-bottom: 1px solid rgba(255,255,255,0.3) !important;
        border-radius: 0 !important;
    }
    
    /* Force dropdown internal text elements to be white */
    div[data-testid="stSelectbox"] * {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
    }

    /* FORM SUBMIT BUTTON (Predict Salary): BLACK BG + WHITE TEXT & BORDER */
    div[data-testid="stFormSubmitButton"] button {
        width: 100% !important;
        background-color: #000000 !important;
        border: 2px solid #ffffff !important;
        padding: 18px 0 !important;
        margin-top: 30px !important;
    }
    
    div[data-testid="stFormSubmitButton"] button * {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        font-family: 'Montserrat', sans-serif !important;
        font-weight: 800 !important;
        text-transform: uppercase !important;
        letter-spacing: 3px !important;
    }

    div[data-testid="stFormSubmitButton"] button:hover {
        background-color: #333333 !important;
    }

    div[data-testid="stForm"] {
        background: rgba(255,255,255,0.015);
        border-top: 1px solid rgba(255,255,255,0.07);
        border-bottom: 1px solid rgba(255,255,255,0.07);
        padding: 50px 20px;
    }

    .result-card {
        border-top: 1px solid rgba(255,255,255,0.08);
        padding: 50px 20px;
        margin-top: 40px;
    }
    .result-text {
        font-family: 'Montserrat', sans-serif;
        font-size: 5vw !important;
        font-weight: 900;
        color: #ffffff;
    }
    </style>
""".replace("BG_PLACEHOLDER", bg_image_base64), unsafe_allow_html=True)

st.markdown("""
<div class="hero-container">
    <div class="hero-main-text">SALARY</div>
    <div class="hero-sub-text">Predictor</div>
</div>
""", unsafe_allow_html=True)

st.markdown('<p class="section-label">Employee Intelligence Engine</p>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Salary Predictor", "Data Insights"])

with tab1:
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            age_input = st.number_input("Age", min_value=18, max_value=70, value=25)
            education_input = st.selectbox("Education Level", ["High School", "Bachelor", "Master", "PhD"], index=1)
            city_input = st.selectbox("City Tier", ["Tier 1", "Tier 2", "Tier 3"], index=0)
        with col2:
            exp_input = st.number_input("Years of Experience", min_value=0, max_value=50, value=3)
            role_input = st.selectbox("Job Role", ["Junior", "Mid", "Senior", "Lead", "Manager"], index=1)
        
        submitted = st.form_submit_button("Predict Salary")

    if submitted:
        if exp_input > (age_input - 18):
            st.error(f"Error: Experience is too high for age {age_input}.")
        else:
            sample_data = {"Age": age_input, "Years_of_Experience": exp_input, "Education_Level": education_input, "Job_Role": role_input, "City_Tier": city_input}
            try:
                salary = predict_salary(sample_data)
                st.markdown(f'<div class="result-card"><p class="section-label">Estimated Annual Package</p><div class="result-text">&#8377;{salary:,.0f}</div></div>', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error: {e}")

with tab2:
    st.markdown('<p class="section-label">Data Insights</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-title">How the data<br>shapes salary</p>', unsafe_allow_html=True)
    
    st.markdown("<p style='font-size:15px; color:#555; margin-top:20px;'>Charts generated from the training pipeline.</p>", unsafe_allow_html=True)
    st.image("plots/salary_by_categories.png", use_container_width=True)
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.image("plots/salary_distribution.png", use_container_width=True, caption="Overall Annual Salary Spread")
    with col_b:
        st.image("plots/correlation_matrix.png", use_container_width=True, caption="Feature Correlation Matrix")