from fastapi import FastAPI 
from classes_values import Identity
import pandas as pd

app = FastAPI()

@app.get('/Saluda')
def roout(name: str): 
    return {'Message': f'Hola soy {name}'}

@app.post('/test')
def testing(info: Identity): 
    return {'Message': f'Hola me llamo {info.name} y mi telefono es {info.phone_number}'}

@app.get('/read_dataframe')
def read_dataframe_properties(): 
    df = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv")
    value = df["sepal_length"]
    return {'Value': value}

@app.get('/read_dataframe/{position}')
def read_dataframe_properties(position: int): 
    df = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv")
    value = df["sepal_length"][position]
    return {'Value': value}