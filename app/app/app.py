from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import sklearn
import pickle
import pandas as pd

class DataIn(BaseModel):
    Active_energy: float
    Mes: int
    Hora: int

with open('app/modelo_isolation_forest.pkl', 'rb') as file:
    model_if = pickle.load(file)

app = FastAPI()

@app.post("/predict/")
async def predict(data: DataIn):
    df = pd.DataFrame(data.dict(), index=[0])
    prediction = model_if.predict(df)
    return {"prediction": str(prediction[0])}

if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=80)