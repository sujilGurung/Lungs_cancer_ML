import os
from contextlib import asynccontextmanager
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

Bin = Literal[0, 1]
state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    bundle = joblib.load("model.joblib")
    state["pipeline"] = bundle["pipeline"]
    state["features"] = bundle["features"]
    yield


app = FastAPI(title="Lung Disease Risk API", version="1.0", lifespan=lifespan)

origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # your Vercel URL(s), set on Render
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",  # any local port
    allow_methods=["*"],
    allow_headers=["*"],
)


class PatientInput(BaseModel):
    AGE: int = Field(ge=1, le=120)
    GENDER: Bin
    SMOKING: Bin
    FINGER_DISCOLORATION: Bin
    MENTAL_STRESS: Bin
    EXPOSURE_TO_POLLUTION: Bin
    LONG_TERM_ILLNESS: Bin
    ENERGY_LEVEL: float = Field(ge=0, le=100)
    IMMUNE_WEAKNESS: Bin
    BREATHING_ISSUE: Bin
    ALCOHOL_CONSUMPTION: Bin
    THROAT_DISCOMFORT: Bin
    OXYGEN_SATURATION: float = Field(ge=50, le=100)
    CHEST_TIGHTNESS: Bin
    FAMILY_HISTORY: Bin
    SMOKING_FAMILY_HISTORY: Bin
    STRESS_IMMUNE: Bin


class Prediction(BaseModel):
    probability: float
    prediction: Literal["YES", "NO"]
    risk_level: Literal["low", "medium", "high"]
    disclaimer: str = "Educational model only. Not medical advice."


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=Prediction)
def predict(data: PatientInput):
    row = pd.DataFrame([data.model_dump()])[state["features"]]  # enforce column order
    p = float(state["pipeline"].predict_proba(row)[0, 1])
    level = "low" if p < 0.33 else "medium" if p < 0.66 else "high"
    return Prediction(
        probability=round(p, 4),
        prediction="YES" if p >= 0.5 else "NO",
        risk_level=level,
    )
