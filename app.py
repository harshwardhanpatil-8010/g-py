from fastapi import FastAPI, HTTPException
from pymongo import MongoClient
import pandas as pd
import joblib
from fastapi.middleware.cors import CORSMiddleware
import logging
# Allow CORS from any origin


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to specific origins for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)


# Load pre-trained model and preprocessors
model = joblib.load("model.pkl")
imputer = joblib.load("imputer.pkl")
scaler = joblib.load("scaler.pkl")

# Connect to MongoDB
client = MongoClient("mongodb+srv://harshwardhanpatil2005:LlnicvQxop7UTW07@grefin-web.ncahj.mongodb.net/?retryWrites=true&w=majority&appName=GREFIN-WEB")
db = client["industry_database"]
collection = db["industry_data"]

# Features used in the model
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

@app.get("/")
def read_root():
    return {"message": "Welcome to the Green Score API"}

@app.get("/favicon.ico")
def favicon():
    return {"message": "Favicon not set"}

def preprocess_input(data: pd.DataFrame):
    # Ensure column names match
    data = data[features]
    data_imputed = pd.DataFrame(imputer.transform(data), columns=features)
    data_scaled = pd.DataFrame(scaler.transform(data_imputed), columns=features)
    return data_scaled

@app.get("/calculate_green_score/{industry}")
def calculate_green_score(industry: str):
    try:
        logging.info(f"Fetching green score for industry: {industry}")
        
        # Normalize the industry name
        industry = industry.strip().lower().replace(" ", "_")

        # Fetch industry data from MongoDB
        industry_data = collection.find_one({"industry": industry})
        if not industry_data:
            logging.error(f"Industry '{industry}' not found in database")
            raise HTTPException(status_code=404, detail=f"Industry '{industry}' not found")

        logging.info(f"Industry data fetched: {industry_data}")

        # Convert MongoDB data to DataFrame
        input_data = pd.DataFrame([industry_data])
        input_data = input_data[features]

        # Preprocess the input data
        input_scaled = preprocess_input(input_data)

        # Predict the green score
        green_score = model.predict(input_scaled)[0]

        # Include necessary fields for the chart
        response = {
            "green_score": green_score,
            "emission": input_data["co2_emissions"].iloc[0],
            "energy_consumption": input_data["energy_consumption_kwh"].iloc[0],
            "waste": input_data["waste_tonnes"].iloc[0],
            "community_impact": input_data["safety_score"].iloc[0], 
            "spend": input_data["operational_spend"].iloc[0],
        }

        logging.info(f"Response generated: {response}")
        return response
    except Exception as e:
        logging.error(f"Error calculating green score: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))