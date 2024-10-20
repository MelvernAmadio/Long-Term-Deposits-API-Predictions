from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd

# Initialize FastAPI app
app = FastAPI()

# Load the trained model and encoders
model = joblib.load('XGB_model.pkl')
encoders = joblib.load('encoders.pkl')

# Define the input data model
class Data(BaseModel):
    age: int
    job: str
    marital: str
    education: str
    default: str
    housing: str
    loan: str
    contact: str
    month: str
    day_of_week: str
    duration: float
    campaign: int
    pdays: int
    previous: int
    poutcome: str


# Define a function to preprocess the input data
def preprocess(data: Data):
    df = pd.DataFrame([data.dict()])

    for column, encoding in encoders.items():
        df.replace(encoding, inplace=True)

    df.fillna(df.mean(), inplace=True)

    return df


# Define the predict endpoint
@app.post("/predict")
def predict(data: Data):
    try:
        # Preprocess the input data
        preprocessed_data = preprocess(data)

        # Make prediction
        prediction = model.predict(preprocessed_data)

        # Return the prediction
        return {"prediction": int(prediction[0])}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))