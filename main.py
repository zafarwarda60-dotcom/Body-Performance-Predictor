"""FastAPI application serving the body-performance prediction models."""
from contextlib import asynccontextmanager
import logging
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
logger = logging.getLogger(__name__)


class Measurements(BaseModel):
    age: float = Field(ge=21, le=64, description="Observed dataset range: 21–64 years")
    gender: Literal["M", "F"]
    height_cm: float = Field(ge=125, le=193.8)
    weight_kg: float = Field(ge=26.3, le=138.1)
    diastolic: float = Field(gt=0, le=126)
    systolic: float = Field(gt=0, le=201)
    grip_force: float = Field(ge=0, le=70.5)
    sit_bend_forward: float = Field(ge=-25, le=213)
    sit_ups_count: float = Field(ge=0, le=80)
    broad_jump_cm: float = Field(ge=0, le=303)
    body_fat: float | None = Field(default=None, ge=3, le=78.4)

    @model_validator(mode="after")
    def blood_pressure_must_be_valid(self):
        if self.systolic <= self.diastolic:
            raise ValueError("systolic must be greater than diastolic")
        return self

    def regression_frame(self):
        return pd.DataFrame([{
            "age": self.age,
            "gender": self.gender,
            "height_cm": self.height_cm,
            "weight_kg": self.weight_kg,
            "diastolic": self.diastolic,
            "systolic": self.systolic,
            "grip_force": self.grip_force,
            "sit_bend_forward": self.sit_bend_forward,
            "sit_ups_count": self.sit_ups_count,
            "broad_jump_cm": self.broad_jump_cm,
        }])

    def classification_frame(self, body_fat):
        frame = self.regression_frame()
        frame["body_fat"] = body_fat
        return frame


@asynccontextmanager
async def lifespan(app: FastAPI):
    regression_path = MODELS / "body_fat_regressor.joblib"
    classification_path = MODELS / "performance_classifier.joblib"
    if not regression_path.exists() or not classification_path.exists():
        raise RuntimeError("Models not found. Run `python train_models.py` first.")
    app.state.regression_model = joblib.load(regression_path)
    app.state.classification_model = joblib.load(classification_path)
    yield


app = FastAPI(title="Body Performance Predictor", version="1.1.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


@app.get("/")
def home():
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict/regression")
def predict_regression(data: Measurements):
    try:
        value = float(app.state.regression_model.predict(data.regression_frame())[0])
        return {"prediction": round(value, 2), "unit": "% body fat"}
    except Exception as exc:
        logger.exception("Body-fat regression prediction failed")
        raise HTTPException(status_code=500, detail="Could not calculate the body-fat estimate.") from exc


@app.post("/predict/classification")
def predict_classification(data: Measurements):
    try:
        body_fat = data.body_fat
        if body_fat is None:
            body_fat = float(app.state.regression_model.predict(data.regression_frame())[0])
        frame = data.classification_frame(body_fat)
        value = str(app.state.classification_model.predict(frame)[0])
        return {
            "prediction": value,
            "label": f"Class {value}",
            "body_fat_used": round(float(body_fat), 2),
            "body_fat_was_estimated": data.body_fat is None,
        }
    except Exception as exc:
        logger.exception("Performance classification prediction failed")
        raise HTTPException(status_code=500, detail="Could not calculate the performance class.") from exc
