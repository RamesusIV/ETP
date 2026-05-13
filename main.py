import joblib
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import logging
app=FastAPI()

model=joblib.load("model.pkl")

logging.basicConfig(filename="prediction.log", level=logging.INFO)

class iris_input(BaseModel):
    sepal_length:float
    sepal_width:float
    petal_length:float
    petal_width:float

@app.get("/")
def home():
    return {"message":"Padharo maare deshhhhhh re~~~~~~~~"}

@app.post("/predict")
def predict(data: iris_input):
    features=np.array([[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]])
    pred=int(model.predict(features)[0])
    logging.info(f"prediction:{pred}")
    return {"prediction": pred}    