# Bank Customer Churn Prediction API 🚀

An end-to-end MLOps pipeline and REST API built with **FastAPI**, **MLflow**, and **Docker** to predict bank customer churn.

---

## 📌 Project Overview
Customer churn is a critical metric for banks. This project builds a machine learning pipeline that predicts whether a customer is likely to leave the bank based on their demographic and financial profile. 

The trained model is logged using **MLflow** and served via a **FastAPI** backend, fully ready for containerized deployment using **Docker**.

---

## 🛠️ Tech Stack & Tools
* **Language:** Python
* **Framework:** FastAPI
* **Data Processing & ML:** Pandas, Scikit-learn
* **Experiment Tracking & Model Registry:** MLflow
* **Containerization:** Docker
* **Version Control:** Git & GitHub

---

## 📁 Repository Structure
```text
churn-mlops/
│
├── data/                    # Dataset folder (Churn_Modelling.csv)
├── mlruns/                  # MLflow experiment tracking logs & artifacts
├── .dockerignore            # Files ignored by Docker build
├── Dockerfile               # Docker configuration for containerization
├── app.py                   # FastAPI application script
├── mlflow.db                # SQLite database for MLflow tracking
├── requirements.txt         # Python dependencies
├── schemas.py               # Pydantic schema for API request validation
└── train.py                 # ML model training and logging script

## How to Run Locally
1. Clone the Repository
git clone [https://github.com/fatima-99-code/churn_mlops.git](https://github.com/fatima-99-code/churn_mlops.git)
cd churn_mlops

2. Set Up Virtual Environment & Install Dependencies
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

pip install -r requirements.txt

3. Run FastAPI Application
uvicorn app:app --reload
(After this run successfully!! Open your browser and visit http://127.0.0.1:8000/docs to test the API endpoints interactively.)

## Running with Docker
Build Docker Image:
docker build -t churn-api .

Run Docker Container:
docker run -p 8000:8000 churn-api
(Access the API at http://localhost:8000.)

