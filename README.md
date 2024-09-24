
# House Price Prediction Using Linear Regression

This project uses a **Machine Learning** algorithm, specifically **Linear Regression**, to predict house prices based on various features such as the average income of an area, the average age of houses in the area, the number of rooms, bedrooms, population, and more.

## Dataset Columns
The dataset used contains the following columns:
- **Avg. Area Income**: Average income of the area where the house is located.
- **Avg. Area House Age**: Average age of houses in the area.
- **Avg. Area Number of Rooms**: Average number of rooms in houses in the area.
- **Avg. Area Number of Bedrooms**: Average number of bedrooms in houses in the area.
- **Area Population**: Population of the area where the house is located.
- **Price**: Price of the house (target variable).

## Prerequisites
To run this project, you will need the following installed:
- Python 3.x
- [Streamlit](https://docs.streamlit.io/)
- Scikit-learn
- Pandas
- Numpy
- Matplotlib

You can install the required libraries by running:

```bash
pip install streamlit scikit-learn pandas numpy matplotlib
**
**Running the Project**
To run this project, follow these steps:

**Clone the repositor**:

git clone https://github.com/yourusername/house-price-prediction.git
cd house-price-prediction

Open your terminal and run the following command:

streamlit run app.py

Open the URL provided by Streamlit in your browser, typically http://localhost:8501.

**Model Overview**
The Linear Regression algorithm is used to create a predictive model that estimates house prices based on the input features.

The features in the dataset are used to train the model, and the target variable is the Price of the house. After training the model, it can predict house prices based on new input data.

**Project Workflow**
**Data Preprocessing**: Cleaning and preparing the data for analysis.
**Exploratory Data Analysis (EDA)**: Visualizing relationships between features.
**Model Training:** Using Linear Regression to build a predictive model.
**Model Evaluation**: Evaluating the model's performance using metrics like R² and Mean Squared Error (MSE).
**Predictions**: Using the model to predict house prices for new data.

**Streamlit App**
I have built an interactive Streamlit app for this project. The app allows users to input new data and get house price predictions based on the trained model.https://housingpricepredictions.streamlit.app/

For a complete guide on using Streamlit to run and deploy the project, refer to the Streamlit Cheatsheet.
**link for viewing my streamlit cheatsheet is below**
https://housingpricepredictions.streamlit.app/

Repository Structure
bash
Copy code
├── app.py                  # Main Streamlit app
├── house_price_model.pkl    # Trained model
├── README.md                # Project documentation
├── requirements.txt         # Required libraries
├── data/                    # Dataset folder
│   └── house_prices.csv     # Dataset file
License
This project is licensed under the MIT License - see the LICENSE file for details.



### Key Features:
- **Overview**: Describes the machine learning algorithm and dataset.
- **Prerequisites**: Includes necessary Python libraries and Streamlit.
- **Running the Project**: Instructions to clone the repo and run the app using Streamlit.
- **Streamlit Link**: A link to the Streamlit cheatsheet for reference.

You can customize the repository URL and any additional details as needed!

