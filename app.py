from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
from typing import Literal

# 1. Initialize FastAPI app
app = FastAPI(title="Churn Prediction API",
              description="An API to predict customer churn.",
              version="1.0")

# 2. Load the trained model and scaler
try:
    model = joblib.load('final_churn_model.pkl')
    scaler = joblib.load('scaler.pkl')
except FileNotFoundError:
    raise RuntimeError("Model or scaler files not found. Please run the notebook cell to save them.")

# 3. Define the input data model for a single prediction
class CustomerFeatures(BaseModel):
    CreditScore: int = Field(..., example=650)
    Geography: Literal['France', 'Germany', 'Spain'] = Field(..., example='France')
    Gender: Literal['Male', 'Female'] = Field(..., example='Male')
    Age: int = Field(..., example=35)
    Tenure: int = Field(..., example=5)
    Balance: float = Field(..., example=120000.0)
    NumOfProducts: int = Field(..., example=1)
    HasCrCard: int = Field(..., example=1, description="1 for Yes, 0 for No")
    IsActiveMember: int = Field(..., example=1, description="1 for Yes, 0 for No")
    EstimatedSalary: float = Field(..., example=100000.0)

# 4. Define the expected column order for the model
expected_columns = [
    'CreditScore', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts',
    'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Balance_to_Salary',
    'Credit_Stability', 'Geography_Germany', 'Geography_Spain'
]

# 5. Create the /predict endpoint
@app.post("/predict")
def predict_churn(features: CustomerFeatures):
    """
    Receives customer data, preprocesses it, and returns a churn prediction.
    """
    # Convert input to a DataFrame
    input_data = pd.DataFrame([features.dict()])

    # --- Preprocessing Pipeline ---
    # Feature Engineering
    input_data['Balance_to_Salary'] = input_data['Balance'] / (input_data['EstimatedSalary'] + 1)
    input_data['Credit_Stability'] = input_data['CreditScore'] / (input_data['Age'] + 1)

    # Categorical Encoding
    input_data['Gender'] = input_data['Gender'].map({'Male': 1, 'Female': 0})
    input_data = pd.get_dummies(input_data, columns=['Geography'], drop_first=False)
    
    # Align columns with the training set
    input_data = input_data.reindex(columns=expected_columns, fill_value=0)

    # Scale numerical features
    features_to_scale = [
        'CreditScore', 'Age', 'Balance', 'EstimatedSalary',
        'Balance_to_Salary', 'Credit_Stability'
    ]
    input_data[features_to_scale] = scaler.transform(input_data[features_to_scale])
    
    # --- Prediction ---
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)

    # Return the result
    return {
        "prediction": int(prediction[0]),
        "prediction_label": "Will Churn" if int(prediction[0]) == 1 else "Will Stay",
        "probability_will_stay": f"{probability[0][0]:.4f}",
        "probability_will_churn": f"{probability[0][1]:.4f}"
    }

@app.get("/")
def read_root():
    return {"message": "Welcome to the Churn Prediction API. Go to /docs to test."}