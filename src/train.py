import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

# Load processed data
X = pd.read_csv("../data/processed/X.csv")
y = pd.read_csv("../data/processed/y.csv")



# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# MLflow Experiment Tracking
mlflow.set_experiment("Loan_Approval_Experiment")

with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train.values.ravel())

    # Predictions & Metrics
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # Log accuracy in MLflow
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(model, "random_forest_model")

    # Save model locally
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "../models/loan_approval_model.pkl")

    print(f"Model trained with accuracy: {acc}")
