import os
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from model import build_and_train_predictor, EplPredictor, DATA_PATH

APP_DIR = Path(__file__).parent
STATIC_DIR = APP_DIR / "static"

app = FastAPI(title="EPL Match Predictor")

# Train model at startup
predictor: Optional[EplPredictor] = None


@app.on_event("startup")
def startup_event() -> None:
    global predictor
    csv_path = os.getenv("EPL_DATA_PATH", DATA_PATH)
    predictor = build_and_train_predictor(csv_path)


@app.get("/api/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/api/teams")
def get_teams() -> Dict[str, List[str]]:
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not available")
    return {"teams": predictor.get_teams()}


@app.post("/api/predict")
def predict(payload: Dict[str, object]) -> Dict[str, object]:
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not available")
    home_team = payload.get("homeTeam")
    away_team = payload.get("awayTeam")
    season = payload.get("season")
    if not isinstance(home_team, str) or not isinstance(away_team, str):
        raise HTTPException(status_code=400, detail="homeTeam and awayTeam must be strings")
    if season is not None and not isinstance(season, int):
        raise HTTPException(status_code=400, detail="season must be an integer if provided")
    result = predictor.predict_match(home_team, away_team, season=season)
    # Add predicted label
    probs = {
        "homeWin": result["homeWin"],
        "draw": result["draw"],
        "awayWin": result["awayWin"],
    }
    predicted_label = max(probs.items(), key=lambda kv: kv[1])[0]
    return {
        "homeTeam": home_team,
        "awayTeam": away_team,
        "season": result["season"],
        "probs": probs,
        "predicted": predicted_label,
    }


# Static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def serve_index() -> FileResponse:
    index_file = STATIC_DIR / "index.html"
    return FileResponse(str(index_file))