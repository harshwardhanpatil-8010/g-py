from fastapi import FastAPI, HTTPException
from pymongo import MongoClient
import pandas as pd
import joblib
from fastapi.middleware.cors import CORSMiddleware
import logging

# Initialize FastAPI app
app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins; restrict for production
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods
    allow_headers=["*"],  # Allows all headers
)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load pre-trained model and preprocessors
model = joblib.load("model.pkl")
imputer = joblib.load("imputer.pkl")
scaler = joblib.load("scaler.pkl")

# Connect to MongoDB
client = MongoClient("MONGO_URI")
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
    """Root endpoint to verify service health"""
    logging.info("Root endpoint accessed")
    return {"message": "Welcome to the Green Score API"}

@app.get("/calculate_green_score/{industry}")
def calculate_green_score(industry: str):
    """Fetch industry data, preprocess it, and calculate the green score"""
    try:
        logging.info(f"Fetching green score for industry: {industry}")
        
        # Normalize the industry name
        industry = industry.strip().lower().replace(" ", "_")

        # Fetch industry data from MongoDB
        industry_data = collection.find_one({"industry": industry})
        if not industry_data:
            logging.error(f"Industry '{industry}' not found in database")
            raise HTTPException(status_code=404, detail=f"Industry '{industry}' not found")

        # Convert MongoDB data to DataFrame
        input_data = pd.DataFrame([industry_data])
        input_data = input_data[features]

        # Preprocess the input data
        input_imputed = pd.DataFrame(imputer.transform(input_data), columns=features)
        input_scaled = pd.DataFrame(scaler.transform(input_imputed), columns=features)

        # Predict the green score
        green_score = model.predict(input_scaled)[0]

        # Return response
        response = {
            "green_score": green_score * 10,  # Scale score for better readability
        }
        logging.info(f"Response generated: {response}")
        return response
    except Exception as e:
        logging.error(f"Error calculating green score: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
