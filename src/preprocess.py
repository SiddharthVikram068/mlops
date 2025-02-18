import pandas as pd
import numpy as np
import joblib
import os

import pymongo
from config import MONGO_URI


# connect to mongo database
def connect_mongo():
    """Connects to MongoDB and returns the database."""
    try:
        client = pymongo.MongoClient(MONGO_URI)
        db = client["general_data"]
        print("connected")
        return db
    except Exception as e:
        print("connection failed")
        return None
    
db = connect_mongo()

train_collection = db["train"]

data = pd.DataFrame(list(train_collection.find()))



# Load dataset
# data = pd.read_csv(input_path)



# Drop Loan_ID (not useful for prediction)
data.drop(columns=["Loan_ID"], inplace=True)

# Handle missing values
data["LoanAmount"].fillna(data["LoanAmount"].median(), inplace=True)
data["Loan_Amount_Term"].fillna(data["Loan_Amount_Term"].median(), inplace=True)
data["Credit_History"].fillna(data["Credit_History"].mode()[0], inplace=True)

# Encode categorical variables
categorical_cols = ["Gender", "Married", "Dependents", "Education", "Self_Employed", "Property_Area"]
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
encoded_features = encoder.fit_transform(data[categorical_cols])
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_cols))

# Save encoder for later use
os.makedirs("models", exist_ok=True)
joblib.dump(encoder, "models/encoder.pkl")

# Drop original categorical columns and add encoded ones
data.drop(columns=categorical_cols, inplace=True)
data = pd.concat([data, encoded_df], axis=1)

# Convert Loan_Status to numerical (1 = Yes, 0 = No)
if "Loan_Status" in data.columns:
    data["Loan_Status"] = data["Loan_Status"].map({"Y": 1, "N": 0})

# Split features and target
X = data.drop(columns=["Loan_Status"])
y = data["Loan_Status"]

# Save processed data
output_dir = "../data/processed"
os.makedirs(output_dir, exist_ok=True)
X.to_csv(f"{output_dir}/X.csv", index=False)
y.to_csv(f"{output_dir}/y.csv", index=False)


