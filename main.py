from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import uvicorn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('drug.csv')
df.head()

x = df[['Age','Sex','BP','Cholesterol','Na_to_K']].values
y = df[['Drug']].values

from sklearn import preprocessing

le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F','M'])
x[:,1] = le_sex.transform(x[:,1])

le_BP = preprocessing.LabelEncoder()
le_BP.fit(['HIGH','LOW','NORMAL'])
x[:,2] = le_BP.transform(x[:,2])

le_chol = preprocessing.LabelEncoder()
le_chol.fit(['HIGH','NORMAL'])
x[:,3] = le_chol.transform(x[:,3])

from sklearn.model_selection import train_test_split

train_x,test_x,train_y,test_y = train_test_split(x,y,test_size = 0.3)

from sklearn.tree import DecisionTreeClassifier as DTC
import sklearn.tree as tree

model = DTC(criterion = 'entropy',max_depth = 4)
model.fit(train_x,train_y)

app = FastAPI()

le_sex = preprocessing.LabelEncoder()
le_BP = preprocessing.LabelEncoder()
le_chol = preprocessing.LabelEncoder()

class PredictionInput(BaseModel):
    Age: int
    Sex: str
    Blood_Pressure: str
    Cholestorl: str
    Na_to_k: float
    
le_sex.fit(['F', 'M'])
le_BP.fit(['HIGH', 'LOW', 'NORMAL'])
le_chol.fit(['HIGH', 'NORMAL'])

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://drug-predition.onrender.com"],
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.post("/predict")
def predict(input_data: PredictionInput):
    features = [[
        input_data.Age,
        le_sex.transform([input_data.Sex])[0],
        le_BP.transform([input_data.Blood_Pressure])[0],
        le_chol.transform([input_data.Cholestorl])[0],
        input_data.Na_to_k
    ]]

    prediction = model.predict(features)
    return {"prediction":str(prediction[0])}
