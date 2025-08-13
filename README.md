EPL Match Predictor

A lightweight FastAPI service that trains on historical Premier League results and serves match outcome probabilities (home win / draw / away win). Includes a minimal front end so anyone can try predictions in the browser.
<img width="1414" height="698" alt="Screenshot 2025-08-13 at 2 24 31 PM" src="https://github.com/user-attachments/assets/8e250787-bf5c-42c0-800a-c84b6bae10a2" />


Backend: FastAPI · Python

ML: pandas · scikit-learn (simple heuristics/counters)

Front end: Static HTML/CSS/JS served by FastAPI

Deploy: Render (free) with Gunicorn/Uvicorn

✨ Features
Trains a predictor at startup from a CSV of EPL matches

REST API endpoints for health, teams, and predict

Minimal UI in /static/index.html to drive the API

Works locally or on Render with one env var: EPL_DATA_PATH

📁 Project structure
php
.
├── main.py                 # FastAPI app (API + static serving)
├── model.py                # EplPredictor + training/helpers
├── requirements.txt
├── Procfile                # process command for Render
├── render.yaml             # Render IaC config
├── static/
│   ├── index.html          # minimal UI
│   └── pl-logo.png         # optional logo
└── data/
    └── epl.csv             # (recommended) training data
If your CSV isn’t at data/epl.csv, set EPL_DATA_PATH to the correct file.

🚀 Quickstart (local)
Python 3.11+ recommended

Create a venv & install deps
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

Point to your dataset (or rename your file to data/epl.csv)
export EPL_DATA_PATH="data/epl.csv"   # Windows: set EPL_DATA_PATH=data/epl.csv

Run
uvicorn main:app --reload --port 8000

Open:
UI → http://127.0.0.1:8000/
