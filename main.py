from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd

app = FastAPI()  # ✅ Cette ligne est cruciale

# Middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Chargement du modèle et des colonnes
model = joblib.load("model.joblib")
numeric_cols, categorical_cols = joblib.load("features.joblib")
all_cols = numeric_cols + categorical_cols

@app.get("/")
def home():
    return {"message": "API de scoring client - prête !"}

@app.post("/predict")
def predict_credit_risk(features: dict):
    try:
        df = pd.DataFrame([features])
        df = df.reindex(columns=all_cols, fill_value=0)
        prob = model.predict_proba(df)[0][1]
        decision = (
            "Éligible" if prob < 0.4 else
            "Risque modéré" if prob < 0.7 else
            "Risque élevé"
        )
        return {"score": round(prob, 4), "decision": decision}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
