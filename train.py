import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
import mlflow
import mlflow.sklearn

# Set MLflow experiment name
mlflow.set_experiment("Bank_Customer_Churn_Experiment")

# Load dataset
df = pd.read_csv("data/Churn_Modelling.csv")

# Drop non-predictive metadata
df = df.drop(columns=["RowNumber", "CustomerId", "Surname"])

# Separate features and target
X = df.drop(columns=["Exited"])
y = df["Exited"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define feature types
categorical_cols = ["Geography", "Gender"]
numerical_cols = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary"]

# Preprocessing Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ]
)

# Full Pipeline
model_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42))
    ]
)

# Run MLflow Tracking
with mlflow.start_run():
    model_pipeline.fit(X_train, y_train)
    predictions = model_pipeline.predict(X_test)
    
    acc = accuracy_score(y_test, predictions)
    prec = precision_score(y_test, predictions)
    rec = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    
    # Log hyperparameters
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 5)
    mlflow.log_param("learning_rate", 0.1)
    
    # Log evaluation metrics
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)
    
    # Log pipeline artifact
    #mlflow.sklearn.log_model(model_pipeline, artifact_path="model")
    # To this:
    mlflow.sklearn.log_model(
    sk_model=model_pipeline, 
    name="model", 
    skops_trusted_types=["xgboost.core.Booster", "xgboost.sklearn.XGBClassifier"]
)


print(f"Training Complete! Accuracy: {acc:.4f}, F1-Score: {f1:.4f}")