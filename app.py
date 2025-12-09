import streamlit as st
import joblib
import numpy as np
import time 


st.set_page_config(
    page_title="Student Score Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)


st.markdown("""
<style>
    /* 1. Global Sizing and Look */
    .stApp { 
        background-color: #f8f8f8; /* Very light gray background */
        color: #333333; 
        font-size: 19px; /* Increased base font size to make min/max digits larger */
    }
    
    /* 2. Main Title Style */
    h1 {
        font-size: 3em !important; 
        color: #1e81b0; /* Modern Blue */
        font-weight: 700;
        padding-top: 10px;
        margin-bottom: 20px;
    }

    /* 3. Input Labels (Slider Heading) - Increased from 1.1em to 1.3em */
    .stSelectbox label, .stSlider label {
        font-size: 1.3em !important; /* Increased size for slider headings */
        font-weight: 600; 
        color: #2c3e50; 
        margin-top: 10px;
    }

    /* 4. Slider Value Display (Current value above handle) - Increased from 1.2em to 1.3em */
    .stSlider > div > div > div:nth-child(2) { 
        font-size: 1.3em !important;
        font-weight: bold; 
        color: #1e81b0; /* Match title color */
    }
    
    /* 5. Primary Button Style */
    .stButton > button {
        height: 3em; 
        font-size: 1.1em; 
        border-radius: 8px; 
        background-color: #1e81b0; 
        color: white;
        border: none;
        font-weight: 600;
    }
    
    /* 6. Metric Value (The final result) */
    [data-testid="stMetricValue"] {
        font-size: 48px !important; 
        color: #2ecc71; /* Success Green for the score */
        font-weight: 700;
    }

</style>
""", unsafe_allow_html=True)


try:
    @st.cache_resource
    def load_model():

        return joblib.load("student_model.pkl")

    model = load_model()
except FileNotFoundError:
    st.error("Error: The model file 'student_model.pkl' was not found.")
    st.stop()
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()


# Define the maps for categorical variables
motivation_map = {"Low": 0, "Medium": 1, "High": 2}
parental_map = {"Low": 0, "Medium": 1, "High": 2}


st.title("🎓 Student Exam Score Prediction System")

# Input Sliders
hours_studied = st.slider("Hours Studied", 0, 20, 6)
attendance = st.slider("Attendance Percentage", 50, 100, 80)
previous_scores = st.slider("Previous Exam Score", 0, 100, 70)

st.divider()

# Input Selectboxes
motivation_str = st.selectbox("Motivation Level", ["Low", "Medium", "High"])
parental_str = st.selectbox("Parental Involvement", ["Low", "Medium", "High"])

# Encoding inputs
motivation_encoded = motivation_map[motivation_str]
parental_encoded = parental_map[parental_str]

st.write("")

if st.button("🔮 Predict Exam Score", use_container_width=True, type="primary"):
    
  
    input_data = np.array([[
        hours_studied, 
        attendance, 
        previous_scores, 
        motivation_encoded, 
        parental_encoded
    ]])
    
    
    with st.spinner('Calculating predicted score...'):
        time.sleep(1) 
        
        #Predict
        try:
            prediction = model.predict(input_data)
            predicted_score = round(float(prediction[0]), 2) 

            st.success("Prediction Complete!")
            
            st.metric(
                label="✅ Predicted Exam Score", 
                value=f"{predicted_score} %"
            )

            if predicted_score > 85:
                st.balloons()
            
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")