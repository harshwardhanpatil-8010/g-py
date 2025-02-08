import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import joblib

# Define the list of features
features = [
    "co2_emissions",
    "energy_consumption_kwh",
    "waste_tonnes",
    "safety_score",
    "employee_satisfaction",
    "compliance_score",
    "violations",
    "operational_spend",
    "season_factor",
    "energy_efficiency_trend",
]

# Function to add a synthetic green_score if not present
def add_green_score(data):
    # Define weights for the features
    weights = {
        "co2_emissions": -0.3,
        "energy_consumption_kwh": -0.2,
        "waste_tonnes": -0.2,
        "safety_score": 0.2,
        "employee_satisfaction": 0.15,
        "compliance_score": 0.25,
        "violations": -0.1,
        "operational_spend": 0.1,
        "season_factor": 0.05,
        "energy_efficiency_trend": 0.15,
    }

    # Compute the green score using a weighted sum
    data["green_score"] = (
        data["co2_emissions"] * weights["co2_emissions"]
        + data["energy_consumption_kwh"] * weights["energy_consumption_kwh"]
        + data["waste_tonnes"] * weights["waste_tonnes"]
        + data["safety_score"] * weights["safety_score"]
        + data["employee_satisfaction"] * weights["employee_satisfaction"]
        + data["compliance_score"] * weights["compliance_score"]
        + data["violations"] * weights["violations"]
        + data["operational_spend"] * weights["operational_spend"]
        + data["season_factor"] * weights["season_factor"]
        + data["energy_efficiency_trend"] * weights["energy_efficiency_trend"]
    )

    # Normalize green_score to fall between 0 and 100
    data["green_score"] = 100 * (data["green_score"] - data["green_score"].min()) / (
        data["green_score"].max() - data["green_score"].min()
    )

    return data

# Function to preprocess the training data
def preprocess_training_data(data):
    # Impute missing values
    imputer = SimpleImputer(strategy="median")
    data[features] = imputer.fit_transform(data[features])

    # Normalize features
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data[features])

    return data_scaled, data["green_score"], imputer, scaler

# Function to train and save the model
def train_and_save_model():
    # Load dataset
    data = pd.read_csv("industry_sustainability_metrics.csv")

    # Add synthetic green_score column if it doesn't exist
    if "green_score" not in data.columns:
        data = add_green_score(data)

    # Preprocess the data
    X, y, imputer, scaler = preprocess_training_data(data)

    # Ensure X is a DataFrame with feature names
    X = pd.DataFrame(X, columns=features)

    # Train the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Save the model and preprocessing objects
    joblib.dump(model, "model.pkl", compress=3)
    joblib.dump(imputer, "imputer.pkl", compress=3)
    joblib.dump(scaler, "scaler.pkl", compress=3 )
    print("Model and preprocessors saved successfully!")

# Train and save the model
train_and_save_model()