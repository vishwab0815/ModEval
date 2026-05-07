from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Any, Dict, List
import os
import io
from src.config import DATA_PATH
from src.preprocessing import load_and_preprocess
from src.feature_engineering import create_features
from src.model_training import TimeSeriesTrainer, save_models, load_best_models
from src.prediction import generate_forecasts
from src.visualization import create_forecast_plot

API_VERSION = "1.0.0"

tags_metadata = [
    {
        "name": "System",
        "description": "Service health, metadata, and available states.",
    },
    {
        "name": "Training",
        "description": "Retrain all models and persist the best state-level models.",
    },
    {
        "name": "Forecasting",
        "description": "Generate 8-week forecasts and visualizations for a specific state.",
    },
]

app = FastAPI(
    title="QuickHyre Forecasting API",
    version=API_VERSION,
    description=(
        "State-level forecasting service for 8-week sales prediction. "
        "The API trains multiple models per state, selects the best performer, "
        "and exposes forecasts and charts through clear JSON endpoints."
    ),
    openapi_tags=tags_metadata,
)


class ForecastPoint(BaseModel):
    date: str
    value: float


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str


class RootResponse(BaseModel):
    message: str
    documentation: str
    version: str
    available_endpoints: Dict[str, str]


class TrainSummary(BaseModel):
    model: str
    rmse: float


class TrainResponse(BaseModel):
    message: str
    total_states: int
    summary: Dict[str, TrainSummary]


class ForecastResponse(BaseModel):
    state: str
    best_model: str
    forecast_dates: List[str]
    forecast_values: List[float]
    forecast: Dict[str, Any]


class StatesResponse(BaseModel):
    total_states: int
    states: List[Dict[str, str]]


def _models_ready() -> bool:
    return os.path.exists(os.path.join("models", "state_best_models.pkl"))


def _load_state_models():
    if not _models_ready():
        raise HTTPException(status_code=400, detail="Models not trained yet. Call POST /train first.")
    return load_best_models()

@app.get("/", response_model=RootResponse, tags=["System"], summary="API landing page")
def root():
    return {
        "message": "QuickHyre Forecasting API is running",
        "documentation": "Open /docs for the interactive Swagger UI or /redoc for the alternative documentation.",
        "version": API_VERSION,
        "available_endpoints": {
            "health": "GET /health",
            "states": "GET /states",
            "train": "POST /train",
            "forecast": "GET /forecast/{state}",
            "plot": "GET /plot/{state}",
        },
    }

@app.get("/health", response_model=HealthResponse, tags=["System"], summary="Check service health")
def health_check():
    return {"status": "healthy", "service": "QuickHyre Forecasting API", "version": API_VERSION}


@app.get("/states", response_model=StatesResponse, tags=["System"], summary="List trained states")
def list_states():
    state_best_models = _load_state_models()
    states = [
        {"state": state, "best_model": info["name"], "accuracy": f"{100 - info['metrics']['MAPE']:.2f}%"}
        for state, info in sorted(state_best_models.items())
    ]
    return {"total_states": len(states), "states": states}

@app.post("/train", response_model=TrainResponse, tags=["Training"], summary="Train and save all models")
def train_pipeline():
    try:
        df = load_and_preprocess(DATA_PATH)
        df = create_features(df)
        trainer = TimeSeriesTrainer(df)
        state_best_models = trainer.train_and_evaluate_all()
        save_models(state_best_models)
        
        report = {
            state: {
                "model": info["name"],
                "rmse": round(info["score"], 2),
            }
            for state, info in state_best_models.items()
        }
        
        return {
            "message": "Training completed successfully",
            "total_states": len(state_best_models),
            "summary": report,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get(
    "/forecast/{state}",
    response_model=ForecastResponse,
    tags=["Forecasting"],
    summary="Get an 8-week forecast for one state",
)
def get_forecast(state: str):
    try:
        state = state.title()

        state_best_models = _load_state_models()
        if state not in state_best_models:
            raise HTTPException(status_code=404, detail=f"State '{state}' not found in trained models.")

        df = load_and_preprocess(DATA_PATH)
        df = create_features(df)

        forecasts = generate_forecasts(state_best_models, df)
        forecast_payload = forecasts[state]
        
        return {
            "state": state,
            "best_model": state_best_models[state]["name"],
            "forecast_dates": forecast_payload["dates"],
            "forecast_values": forecast_payload["forecast"],
            "forecast": forecast_payload,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/plot/{state}", tags=["Forecasting"], summary="Render a forecast plot as PNG")
def get_plot(state: str):
    try:
        state = state.title()

        state_best_models = _load_state_models()
        if state not in state_best_models:
            raise HTTPException(status_code=404, detail=f"State '{state}' not found.")

        df = load_and_preprocess(DATA_PATH)
        forecasts = generate_forecasts(state_best_models, df)

        plot_buf = create_forecast_plot(df, state, state_best_models, forecasts[state])
        
        return StreamingResponse(plot_buf, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
