from fastapi import FastAPI
import pandas as pd
import mlflow.sklearn
from schemas import ChurnInput

app = FastAPI(title="Bank Customer Churn Prediction API")

# Load trained pipeline directly from your local mlruns folder
model = mlflow.sklearn.load_model("mlruns/1/models/m-72252649779547f79be5850a39bfbba1/artifacts")

@app.get("/")
def home():
    return {"message": "Customer Churn Prediction API is running!"}

@app.post("/predict")
def predict(data: ChurnInput):
    input_df = pd.DataFrame([data.dict()])
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    
    return {
        "churn_prediction": int(prediction),
        "churn_probability": float(probability),
        "status": "Churn Risk" if prediction == 1 else "Retained"
    }