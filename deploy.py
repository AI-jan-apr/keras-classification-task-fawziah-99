from fastapi import FastAPI
from pydantic import BaseModel, conlist
import numpy as np
import pickle


app = FastAPI()


# Load model and scaler
with open("model_weights.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler_weights.pkl", "rb") as f:
    scaler = pickle.load(f)


class Features(BaseModel):
    features: conlist(float, min_length=30, max_length=30)


@app.get("/")
def read_root():
    return {"message": "API is running"}


@app.post("/predict")
def predict(data: Features):
    
    input_data = np.array(data.features).reshape(1, -1)

    scaled_data = scaler.transform(input_data)

    prediction = model.predict(scaled_data)
    
    predicted_class = int((prediction > 0.5)[0][0])
    
    return {"prediction": predicted_class,}