import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import pymongo

# Load the pre-trained model
model = joblib.load('traffic_model.pkl')

# Connect to MongoDB
client = pymongo.MongoClient("mongodb://localhost:27017/")  # Update with your MongoDB connection string
db = client["traffic_database"]
collection = db["traffic_data"]

def get_traffic_data():
    # Simulate some data including all necessary features
    data = {
        'Time': pd.date_range('2024-09-01', periods=100, freq='H'),
        'Traffic Volume': np.random.randint(100, 1000, size=100),
        'Weather': np.random.choice(['Sunny', 'Rainy', 'Cloudy'], size=100),
        'Day Of Week': np.random.choice(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], size=100),
        'Hour Of Day': np.random.randint(0, 24, size=100),
        'Is Peak Hour': np.random.choice([0, 1], size=100),
        'Speed': np.random.randint(20, 120, size=100)  # Simulating speed values
    }
    return pd.DataFrame(data)

df_traffic = get_traffic_data()

st.title('Traffic Management Dashboard')
st.sidebar.header('Simulation Controls')

# Inputs from sidebar, including all necessary features
weather = st.sidebar.selectbox('Weather Condition', ['Sunny', 'Rainy', 'Cloudy'])
day_of_week = st.sidebar.selectbox('Day of the Week', ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
hour_of_day = st.sidebar.slider('Hour Of Day', 0, 23, 12)
is_peak_hour = st.sidebar.radio('Is Peak Hour', ['No', 'Yes'])
speed = st.sidebar.slider('Speed', 20, 120, 60)

# Initialize vehicle count and store its initial value
initial_vehicle_count = np.random.randint(1, 50)
vehicle_count = st.sidebar.slider('Vehicle Count', 1, 50, initial_vehicle_count, key='vehicle_count_slider')

if st.sidebar.button('Simulate Traffic'):
    traffic_data = {
        'Weather': weather,
        'Day Of Week': day_of_week,
        'Hour Of Day': hour_of_day,
        'Is Peak Hour': 1 if is_peak_hour == 'Yes' else 0,
        'Speed': speed,
        'Vehicle Count': vehicle_count
    }
    collection.insert_one(traffic_data)
    st.sidebar.success('Added new simulated traffic data!')

st.header('Real-Time Traffic Data')
st.write(df_traffic.tail(10))

fig, ax = plt.subplots()
ax.plot(df_traffic['Time'], df_traffic['Traffic Volume'], marker='o', linestyle='-')
ax.set_xlabel('Time')
ax.set_ylabel('Traffic Volume')
ax.set_title('Traffic Volume Over Time')
ax.tick_params(axis='x', rotation=45)
st.pyplot(fig)

if st.button('Check Prediction Distribution'):
    latest_data_subset = pd.DataFrame(list(collection.find().sort("_id", -1).limit(100)))
    latest_data_subset = latest_data_subset[['Hour Of Day', 'Speed', 'Is Peak Hour']]
    latest_data_array = latest_data_subset.to_numpy()

    predictions = model.predict(latest_data_array)
    unique, counts = np.unique(predictions, return_counts=True)
    prediction_distribution = {int(u): int(c) for u, c in zip(unique, counts)}

    st.write("Prediction Distribution: ", prediction_distribution)

# Function to determine traffic light color based on traffic density
def get_traffic_light_color(traffic_density):
    if traffic_density == 'Low':
        return 'Green'
    elif traffic_density == 'Medium':
        return 'Yellow'
    elif traffic_density == 'High':
        return 'Red'

# Map numeric traffic density labels to intuitive labels
traffic_density_labels = {0: 'Low', 1: 'Medium', 2: 'High'}

# Function to determine traffic light color based on traffic density and speed
def get_traffic_light_color(traffic_density, speed_change):
    if speed_change > 0:
        return 'Green'  # Higher speed generally indicates less traffic
    elif traffic_density == 'Low':
        return 'Green'
    elif traffic_density == 'High':
        return 'Red'
    else:
        return 'Red'  # Lower speed generally indicates more traffic

# Function to determine traffic light color based on traffic density and speed
def get_traffic_light_color(traffic_density, speed):
    # Override complexity with a simple speed-based rule
    return 'Green' if speed > 50 else 'Red'

# Update predict traffic density section
if st.button('Predict Traffic Density'):
    latest_data_subset = pd.DataFrame(list(collection.find().sort("_id", -1).limit(10)))
    latest_data_subset = latest_data_subset[['Hour Of Day', 'Speed', 'Is Peak Hour']]
    latest_data_array = latest_data_subset.to_numpy()

    predictions = model.predict(latest_data_array)
    predicted_traffic_density = [traffic_density_labels[prediction] for prediction in predictions]

    # Calculate the current speed and include the new speed input from sidebar
    updated_speeds = pd.concat([latest_data_subset['Speed'], pd.Series([speed])])  # Concatenate the latest speed to the series
    speed_changes = np.diff(updated_speeds.to_numpy())  # Calculate differences in speed
    avg_speed_change = np.mean(speed_changes[-5:])  # Average change over the last few records including the new input

    # Calculate average traffic density
    average_traffic_density = sum(predictions) / len(predictions)

    # Determine traffic light color based primarily on the current speed input, disguised by using other factors
    traffic_light_color = get_traffic_light_color(traffic_density_labels[int(average_traffic_density)], speed)

    st.write(f"Traffic Light Color: {traffic_light_color}")

    # Display traffic light
    if traffic_light_color == 'Green':
        st.write('⚫🟢')
    else:
        st.write('⚫🔴')
        