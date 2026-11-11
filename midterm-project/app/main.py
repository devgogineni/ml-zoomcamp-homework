import pickle
from contextlib import asynccontextmanager
from datetime import date as Date
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, create_model, model_validator

from training import MODEL_PATH, engineer_features

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

# numeric model inputs: bounds are physical limits, defaults are the training medians
FIELDS = [
    {"name": "tempmax", "label": "Max temperature", "unit": "°C", "min": -50, "max": 60, "step": 0.1, "default": 15.1},
    {"name": "tempmin", "label": "Min temperature", "unit": "°C", "min": -50, "max": 60, "step": 0.1, "default": 8.3},
    {"name": "temp", "label": "Mean temperature", "unit": "°C", "min": -50, "max": 60, "step": 0.1, "default": 11.9},
    {"name": "dew", "label": "Dew point", "unit": "°C", "min": -50, "max": 40, "step": 0.1, "default": 7.7},
    {"name": "humidity", "label": "Humidity", "unit": "%", "min": 0, "max": 100, "step": 0.1, "default": 77.6},
    {"name": "precip", "label": "Precipitation", "unit": "mm", "min": 0, "max": 500, "step": 0.1, "default": 0},
    {"name": "precipprob", "label": "Precipitation probability", "unit": "%", "min": 0, "max": 100, "step": 1, "default": 0},
    {"name": "precipcover", "label": "Precipitation cover", "unit": "%", "min": 0, "max": 100, "step": 0.1, "default": 0},
    {"name": "cloudcover", "label": "Cloud cover", "unit": "%", "min": 0, "max": 100, "step": 0.1, "default": 56.7},
]

# every training day has exactly one sky condition plus optional rain and/or snow
SKY = {"clear": "Clear", "partially_cloudy": "Partially cloudy", "overcast": "Overcast"}
PRECIPITATION = {"rain": "Rain", "snow": "Snow"}

WeatherInput = create_model(
    "WeatherInput",
    **{f["name"]: (float, Field(ge=f["min"], le=f["max"], description=f["label"])) for f in FIELDS},
)


class PredictionRequest(WeatherInput):
    date: Date = Field(description="Only the day of the year is used by the model")
    sky: Literal["clear", "partially_cloudy", "overcast"]
    precipitation: list[Literal["rain", "snow"]] = []

    @model_validator(mode="after")
    def check_temperatures(self):
        if not self.tempmin <= self.temp <= self.tempmax:
            raise ValueError("temperatures must satisfy tempmin <= temp <= tempmax")
        return self


class PredictionResponse(BaseModel):
    solar_energy_generated: float


model_bundle = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    with open(PROJECT_DIR / MODEL_PATH, "rb") as f_in:
        model_bundle["dv"], model_bundle["scaler"], model_bundle["model"] = pickle.load(f_in)
    yield


app = FastAPI(title="Solar energy generation predictor", lifespan=lifespan)


def predict(request: PredictionRequest) -> float:
    conditions = [SKY[request.sky]] + [PRECIPITATION[p] for p in request.precipitation]
    row = request.model_dump(exclude={"sky", "precipitation"})
    row["date"] = pd.Timestamp(request.date)
    row["conditions"] = ", ".join(conditions)

    # reuse the training feature engineering so serving matches training exactly
    df = engineer_features(pd.DataFrame([row]))
    dv = model_bundle["dv"]
    record = df[list(dv.feature_names_)].to_dict(orient="records")

    X = model_bundle["scaler"].transform(dv.transform(record))
    # a linear model can go below zero, but generated energy cannot
    return max(float(model_bundle["model"].predict(X)[0]), 0.0)


@app.get("/api/features")
def features():
    """Describes the form inputs so the frontend stays in sync with the model."""
    return {
        "fields": FIELDS,
        "sky": [{"value": k, "label": v} for k, v in SKY.items()],
        "precipitation": [{"value": k, "label": v} for k, v in PRECIPITATION.items()],
    }


@app.post("/api/predict", response_model=PredictionResponse)
def predict_endpoint(request: PredictionRequest):
    return PredictionResponse(solar_energy_generated=round(predict(request), 2))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(BASE_DIR / "static" / "index.html")


app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
