import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv('cStick.csv')  # Replace with the correct path
y = df['Accelerometer']
x = df.drop('Accelerometer', axis=1)

# Split the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)

# Train the model with higher depth
rf = RandomForestRegressor(max_depth=10, random_state=100)  # Increased max_depth to 10 for better learning
rf.fit(x_train, y_train)

# Inject CSS for styling (same as before)
st.markdown("""
    <style>
        body {
            background-color: #f9fafb;
            font-family: 'Arial', sans-serif;
        }
        .title {
            font-size: 28px;
            font-weight: bold;
            color: #4338ca;
            text-align: center;
            margin-bottom: 20px;
        }
        .card {
            background-color: #ffffff;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.1);
        }
        .input-section {
            margin-bottom: 20px;
        }
        .predict-btn {
            background-color: #4338ca;
            color: white;
            padding: 10px 20px;
            border-radius: 5px;
            border: none;
            font-size: 18px;
            cursor: pointer;
            margin-top: 20px;
        }
        .predict-btn:hover {
            background-color: #3730a3;
        }
        .result {
            text-align: center;
            font-size: 24px;
            font-weight: bold;
            margin-top: 20px;
        }
        .fall {
            color: #ef4444;
        }
        .stable {
            color: #10b981;
        }
        .already-fallen {
            color: #d97706;
        }
    </style>
""", unsafe_allow_html=True)

# Streamlit interface
st.markdown("<div class='title'>Fall Detection Prediction</div>", unsafe_allow_html=True)

st.markdown("<div class='card'>", unsafe_allow_html=True)
st.write("""
### Input the sensor data to predict if the person will fall, is stable, or has already fallen.
""")
st.markdown("</div>", unsafe_allow_html=True)

# Input fields for sensor data
st.markdown("<div class='card input-section'>", unsafe_allow_html=True)
Distance = st.number_input('Distance', min_value=0.0, max_value=100.0, value=50.0)
Pressure = st.number_input('Pressure', min_value=0.0, max_value=100.0, value=50.0)
HRV = st.number_input('HRV', min_value=0.0, max_value=100.0, value=50.0)
Sugar_level = st.number_input('Sugar level', min_value=0.0, max_value=100.0, value=50.0)
SpO2 = st.number_input('SpO2', min_value=0.0, max_value=100.0, value=50.0)
Decision = st.number_input('Decision ', min_value=0.0, max_value=100.0, value=50.0)
st.markdown("</div>", unsafe_allow_html=True)

# Create a new data point based on the input
new_data = pd.DataFrame({
    'Distance': [Distance],
    'Pressure': [Pressure],
    'HRV': [HRV],
    'Sugar level': [Sugar_level],
    'SpO2': [SpO2],
    'Decision ': [Decision],
})

# Predict using the model
if st.button('Predict', key='predict_btn'):
    prediction = rf.predict(new_data)
    
    # Debug: Print prediction values to see actual output
    st.write(f"Prediction Value: {prediction[0]}")

    # Adjusting the thresholds based on prediction ranges
    if prediction[0] >= 1:  # Set more meaningful threshold values after analyzing your data
        st.markdown("<p class='fall'>Prediction: The person is likely to fall.</p>", unsafe_allow_html=True)
    elif 0.5 <= prediction[0] <= 1.5:
        st.markdown("<p class='stable'>Prediction: The person is stable.</p>", unsafe_allow_html=True)
    else:
        st.markdown("<p class='already-fallen'>Prediction: The person has already fallen.</p>", unsafe_allow_html=True)

st.write("Developed by Devank Gupta")
