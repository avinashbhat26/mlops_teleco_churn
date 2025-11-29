from fastapi import FastAPI
from pydantic import BaseModel

from src.predict import ChurnModelService

app = FastAPI(title="Telco Churn Prediction API")
model_service = ChurnModelService()


class CustomerFeatures(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    InternetService: str
    Contract: str
    MonthlyCharges: float
    TotalCharges: float


@app.get("/")
def root():
    return {"message": "Telco Churn API is running"}


@app.post("/predict")
def predict_churn(features: CustomerFeatures):
    result = model_service.predict_single(features.dict())
    return result
