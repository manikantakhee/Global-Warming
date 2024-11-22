import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
import seaborn as sns

# Read data from CSV file
data = pd.read_csv("data.csv")

# Preprocessing pipeline
numeric_features = [
    "Load Demand (kW)",
    "Voltage (V)",
    "Initial Cost (USD)",
    "Maintenance Cost (USD)",
]
categorical_features = ["Location"]

numeric_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(drop="first")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# Define models
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
lr_regressor = LinearRegression()
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)

# Train models (using the loaded data)
X = data.drop(columns=["Energy Source", "Transition Cost (USD)"])
y_source = data["Energy Source"]
y_cost = data["Transition Cost (USD)"]

# Fit preprocessors and models
X_processed = preprocessor.fit_transform(X)

rf_classifier.fit(X_processed, y_source)
lr_regressor.fit(X_processed, y_cost)
rf_regressor.fit(X_processed, y_cost)

# Streamlit UI
st.title("Global Warming: Awareness and Solutions")

st.markdown("Guided by - Dr. Arun Kumar Sir")
st.markdown("21BAI1730-21BAI1488-21BAI1734-21BRS1441")


# Spacer to create a scroll effect before prediction inputs
st.markdown("<br><br><br>", unsafe_allow_html=True)

# Input fields for the prediction
location = st.selectbox("Location", ["Coastal", "Mountain", "Urban"])
load_demand = st.number_input("Load Demand (kW)", min_value=1, max_value=100)
voltage = st.number_input("Voltage (V)", min_value=100, max_value=1000)
initial_cost = st.number_input("Initial Cost (USD)", min_value=1000, max_value=50000)
maintenance_cost = st.number_input(
    "Maintenance Cost (USD)", min_value=1000, max_value=10000
)

# When the user clicks the button to get predictions
if st.button("Get Prediction"):
    # Prepare the input data for the model
    input_data = pd.DataFrame(
        {
            "Location": [location],
            "Load Demand (kW)": [load_demand],
            "Voltage (V)": [voltage],
            "Initial Cost (USD)": [initial_cost],
            "Maintenance Cost (USD)": [maintenance_cost],
        }
    )

    input_data_processed = preprocessor.transform(input_data)

    # Predictions
    energy_source_prediction = rf_classifier.predict(input_data_processed)
    transition_cost_lr_prediction = lr_regressor.predict(input_data_processed)
    transition_cost_rf_prediction = rf_regressor.predict(input_data_processed)

    # Display predictions
    st.subheader("Predictions")
    st.write(f"Predicted Energy Source: {energy_source_prediction[0]}")
    st.write(
        f"Predicted Transition Cost (USD) - Linear Regression: {transition_cost_lr_prediction[0]:.2f}"
    )
    st.write(
        f"Predicted Transition Cost (USD) - Random Forest: {transition_cost_rf_prediction[0]:.2f}"
    )

    # Model Metrics
    accuracy = rf_classifier.score(X_processed, y_source)  # Accuracy for classifier
    mae_lr = np.mean(
        np.abs(y_cost - lr_regressor.predict(X_processed))
    )  # MAE for Linear Regression
    mae_rf = np.mean(
        np.abs(y_cost - rf_regressor.predict(X_processed))
    )  # MAE for Random Forest

    
