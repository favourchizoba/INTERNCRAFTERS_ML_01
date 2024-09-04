import streamlit as st
import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import joblib 

# Load the housing dataset
data = pd.read_csv('housing.csv')
st.markdown("<h1 style = 'color: #1d267d; text-align: center; font-size: 60px; font-family: Georgia'>HOUSE PRICE PREDICTION</h1>", unsafe_allow_html = True)


st.markdown("<br>", unsafe_allow_html=True)

# #add image
st.image('pngwing.com (28).png',width = 500)
#add sidebar image
st.sidebar.image('pngwing.com (30).png', width=300, caption = 'Welcome User')


# Add divider and spacing
st.sidebar.divider()

# Add a header for project background information with styled divider
st.write(
    f"<h2 style='border-bottom:2px solid green; padding-bottom:10px;'>Project Background Information</h2>", 
    unsafe_allow_html=True
)


 #for display theme 
primaryColor="#FF4B4B"
backgroundColor="#70E6D2"
secondaryBackgroundColor="#B4F2E8"
textColor="#31333F"
font="serif"


# Write project background information
st.write("The primary objective of this predictive model is to analyze housing data using machine learning algorithms. By leveraging demographic, socio-economic, and housing-related features, the model aims to predict housing prices accurately. This information can be valuable for various stakeholders, including home buyers, real estate agents, and policymakers.")

# Add spacing using Markdown
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)


# drop the 'Street' column from the dataset
data = data.drop('Address', axis=1)

# dataset Overview
st.write(
    f"<h2 style='border-bottom:2px solid green; padding-bottom:10px;'>dataset Overview</h2>", 
    unsafe_allow_html=True
)

if st.checkbox('Show Raw data'):
    st.write(data)
sel_cols = ['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
            'Avg. Area Number of Bedrooms', 'Area Population']

# # Function to get user data
def get_user_input():
    input_your_choice = st.sidebar.radio('Select Your Input Type', ['Slider Input', 'Number Input'])
    
    if input_your_choice == 'Slider Input':
        area_income = st.sidebar.slider('Average Area Income', float(data['Avg. Area Income'].min()), float(data['Avg. Area Income'].max()))
        house_age = st.sidebar.slider('Average House Age', float(data['Avg. Area House Age'].min()), float(data['Avg. Area House Age'].max()))
        room_num = st.sidebar.slider('Average Number of Rooms', float(data['Avg. Area Number of Rooms'].min()), float(data['Avg. Area Number of Rooms'].max()))
        bedrooms = st.sidebar.slider('Average Number of Bedrooms', float(data['Avg. Area Number of Bedrooms'].min()), float(data['Avg. Area Number of Bedrooms'].max()))
        population = st.sidebar.slider('Area Population', float(data['Area Population'].min()), float(data['Area Population'].max()))
    else:
        area_income = st.sidebar.number_input('Average Area Income', float(data['Avg. Area Income'].min()), float(data['Avg. Area Income'].max()))
        house_age = st.sidebar.number_input('Average House Age', float(data['Avg. Area House Age'].min()), float(data['Avg. Area House Age'].max()))
        room_num = st.sidebar.number_input('Average Number of Rooms', float(data['Avg. Area Number of Rooms'].min()), float(data['Avg. Area Number of Rooms'].max()))
        bedrooms = st.sidebar.number_input('Average Number of Bedrooms', float(data['Avg. Area Number of Bedrooms'].min()), float(data['Avg. Area Number of Bedrooms'].max()))
        population = st.sidebar.number_input('Area Population', float(data['Area Population'].min()), float(data['Area Population'].max()))
    
    user_input = {
        'Avg. Area Income': area_income,
        'Avg. Area House Age': house_age,
        'Avg. Area Number of Rooms': room_num,
        'Avg. Area Number of Bedrooms': bedrooms,
        'Area Population': population
    }
    
    features = pd.DataFrame(user_input, index=[0])
    return features

# Get user input
user_input = get_user_input()



st.markdown("<br>", unsafe_allow_html= True)
st.divider()
st.subheader('Users Inputs')
st.dataframe(user_input, use_container_width = True)


# Load the scalers and model
area_population_scaler = joblib.load('Area Population_scaler.pkl')
area_income_scaler = joblib.load('Avg. Area Income_scaler.pkl')
model = joblib.load('HousepriceModel.pkl')

# Transform user input
user_input['Area Population'] = area_population_scaler.transform(user_input[['Area Population']])
user_input['Avg. Area Income'] = area_income_scaler.transform(user_input[['Avg. Area Income']])

# st.header('Transformed Input Variable')
# st.dataframe(user_input, use_container_width = True)


# Predict house price
predicted = model.predict(user_input)


# Tabs for prediction 


#to have a button for the user
if st.button('Predict Price'):
    predicted_price = model.predict(user_input)
    st.success(f"The Price of this House is  {predicted_price[0].round()}")
    


# User Guide and Help Section
st.header('User Guide & Help')

if st.checkbox('Show User Guide'):
    st.subheader('User Guide')
    st.write("""
    - Use the sliders or number inputs to provide average area income, house age, number of rooms, bedrooms, and population.
    - Click 'Predict Price' to see the predicted house price.
    - Check 'Show Raw Data'to explore the dataset.
    """)

    st.subheader('Need Help?')
    st.write("""
    - If you encounter any issues or have questions, please contact our support team at chibuezechizobafavour@gmail.com
    """)

