from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import numpy as np
import pandas as pd
import uvicorn
import dill
import os

from pydantic import BaseModel

# Setup FastAPI and Jinja2 templates
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Mount static if needed (for CSS, JS)
# app.mount("/static", StaticFiles(directory="static"), name="static")

# Load model and preprocessor
model_path = os.path.join("artifacts", "model.pkl")
preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

with open(model_path, "rb") as f:
    model = dill.load(f)
with open(preprocessor_path, "rb") as f:
    preprocessor = dill.load(f)

# Home route
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

# Prediction route
@app.post("/predictdata", name="predict_datapoint")
async def predict_data(
    request: Request,
    gender: str = Form(...),
    ethnicity: str = Form(...),
    parental_level_of_education: str = Form(...),
    lunch: str = Form(...),
    test_preparation_course: str = Form(...),
    writing_score: float = Form(...),
    reading_score: float = Form(...),
):
    try:
        data_dict = {
            "gender": [gender],
            "race_ethnicity": [ethnicity],
            "parental_level_of_education": [parental_level_of_education],
            "lunch": [lunch],
            "test_preparation_course": [test_preparation_course],
            "writing_score": [writing_score],
            "reading_score": [reading_score],
        }
        df = pd.DataFrame(data_dict)

        transformed = preprocessor.transform(df)
        prediction = model.predict(transformed)[0]

        return templates.TemplateResponse("home.html", {
            "request": request,
            "results": round(prediction, 2)
        })
    except Exception as e:
        return templates.TemplateResponse("home.html", {
            "request": request,
            "error": str(e)
        })