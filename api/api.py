from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import traceback

# ======================================================
# CHARGEMENT DU PIPELINE (IMPORTANT)
# ======================================================

MODEL_PATH = r"C:\Users\DC\Projects\Predict_Sales\models\sales_pipeline.pkl"

pipeline = joblib.load(MODEL_PATH)

print("✅ Pipeline chargé avec succès !")

# ======================================================
# API
# ======================================================

app = FastAPI(
    title="API Prédiction des Ventes",
    description="API ML PRO (Pipeline sklearn)",
    version="2.0"
)

# ======================================================
# INPUT (DOIT MATCH TRAINING)
# ======================================================

class DonneesCommande(BaseModel):

    PRODUCTLINE: str
    DEALSIZE: str
    COUNTRY: str
    STATUS: str

    MONTH: int
    YEAR: int

    QUARTER: int
    FIN_ANNEE: int
    SAISON: str

    SALES_MOYEN_CLIENT: float
    SALES_TOTAL_CLIENT: float
    SALES_MOYEN_PRODUIT: float
    SALES_MOYEN_DEALSIZE: float
    SALES_MOYEN_PAYS: float

# ======================================================
# FEATURE ENGINEERING (OPTIONNEL MAIS PROPRE)
# ======================================================

def ajouter_features(df):

    df = df.copy()

    # sécurise si Streamlit n’envoie pas ces champs
    if "QUARTER" not in df:
        df["QUARTER"] = ((df["MONTH"] - 1) // 3) + 1

    if "FIN_ANNEE" not in df:
        df["FIN_ANNEE"] = df["MONTH"].apply(lambda x: 1 if x in [10,11,12] else 0)

    if "SAISON" not in df:
        df["SAISON"] = df["MONTH"].apply(
            lambda x:
                "Hiver" if x in [12,1,2]
                else "Printemps" if x in [3,4,5]
                else "Ete" if x in [6,7,8]
                else "Automne"
        )

    return df

# ======================================================
# ROUTE TEST
# ======================================================

@app.get("/")
def home():
    return {"message": "API PRO Ventes - Pipeline OK"}

# ======================================================
# PREDICTION
# ======================================================

@app.post("/predire")
def predict(data: DonneesCommande):

    try:

        # 1. DataFrame
        df = pd.DataFrame([data.dict()])

        # 2. Feature engineering sécurité
        df = ajouter_features(df)

        # 3. Prédiction pipeline (ULTRA SIMPLE)
        pred_log = pipeline.predict(df)
        pred = np.expm1(pred_log)

        return {
            "prediction_ventes": round(float(pred[0]), 2),
            "unite": "dollars ($)"
        }

    except Exception as e:

        erreur = traceback.format_exc()
        print("❌ ERREUR API :", erreur)

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )