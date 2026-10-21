
from pathlib import Path
from datetime import date
import logging

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ======================================================
# 1. CONFIGURATION
# ======================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "models" / "sales_pipeline.pkl"
DATA_PATH = ROOT_DIR / "data" / "Auto Sales data.csv"

if not MODEL_PATH.is_file():
    raise FileNotFoundError(f"Modèle introuvable : {MODEL_PATH}")

if not DATA_PATH.is_file():
    raise FileNotFoundError(f"Dataset introuvable : {DATA_PATH}")

pipeline = joblib.load(MODEL_PATH)

# Charger les données historiques
historique = pd.read_csv(DATA_PATH)
# Le CSV source utilise le format jour/mois/année (ex. 24/02/2018).
historique["ORDERDATE"] = pd.to_datetime(
    historique["ORDERDATE"], format="%d/%m/%Y", errors="coerce"
)
historique["SALES"] = pd.to_numeric(
    historique["SALES"], errors="coerce"
)

historique = historique.dropna(
    subset=["ORDERDATE", "SALES"]
).copy()

logger.info("Pipeline chargé depuis %s", MODEL_PATH)
logger.info("Historique chargé : %s lignes", len(historique))

# Les noms et l'ordre doivent correspondre.
FEATURE_COLS = [
    "PRODUCTLINE",
    "DEALSIZE",
    "COUNTRY",
    "STATUS",
    "SAISON",
    "MONTH",
    "YEAR",
    "QUARTER",
    "FIN_ANNEE",
    "SALES_MOYEN_CLIENT",
    "SALES_TOTAL_CLIENT",
    "SALES_MOYEN_PRODUIT",
    "SALES_MOYEN_DEALSIZE",
    "SALES_MOYEN_PAYS",
]

# ======================================================
# 2. APPLICATION FASTAPI
# ======================================================

app = FastAPI(
    title="API de prédiction des ventes",
    description=(
        "Prédiction des ventes avec calcul automatique "
        "des statistiques historiques."
    ),
    version="3.0.0",
)

# ======================================================
# 3. DONNÉES D'ENTRÉE
# ======================================================

class DonneesCommande(BaseModel):
    CUSTOMERNAME: str = Field(min_length=1)
    ORDERDATE: date

    PRODUCTLINE: str = Field(min_length=1)
    DEALSIZE: str = Field(min_length=1)
    COUNTRY: str = Field(min_length=1)
    STATUS: str = Field(min_length=1)

# ======================================================
# 4. VARIABLES TEMPORELLES
# ======================================================

def ajouter_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    mois = df["MONTH"]

    df["QUARTER"] = ((mois - 1) // 3) + 1
    df["FIN_ANNEE"] = mois.isin([10, 11, 12]).astype(int)

    saisons = {
        12: "Hiver", 1: "Hiver", 2: "Hiver",
        3: "Printemps", 4: "Printemps", 5: "Printemps",
        6: "Ete", 7: "Ete", 8: "Ete",
        9: "Automne", 10: "Automne", 11: "Automne",
    }

    df["SAISON"] = mois.map(saisons)
    return df

# ======================================================
# 5. CALCUL AUTOMATIQUE DES STATISTIQUES HISTORIQUES
# ======================================================

def calculer_stats_commande(
    nom_client: str,
    date_commande: pd.Timestamp,
    productline: str,
    dealsize: str,
    country: str,
) -> dict:
    # Exclure la date de prédiction et toutes les dates futures.
    passe = historique[
        historique["ORDERDATE"] < date_commande
    ].copy()

    # Statistiques du client
    ventes_client = passe[
        passe["CUSTOMERNAME"].astype(str).str.strip().str.casefold()
        == nom_client.strip().casefold()
    ]["SALES"]

    # Statistiques par catégorie
    ventes_produit = passe.loc[
        passe["PRODUCTLINE"] == productline, "SALES"
    ]

    ventes_taille = passe.loc[
        passe["DEALSIZE"] == dealsize, "SALES"
    ]

    ventes_pays = passe.loc[
        passe["COUNTRY"] == country, "SALES"
    ]

    return {
        "SALES_MOYEN_CLIENT": (
            float(ventes_client.mean())
            if not ventes_client.empty else 0.0
        ),
        "SALES_TOTAL_CLIENT": (
            float(ventes_client.sum())
            if not ventes_client.empty else 0.0
        ),
        "SALES_MOYEN_PRODUIT": (
            float(ventes_produit.mean())
            if not ventes_produit.empty else 0.0
        ),
        "SALES_MOYEN_DEALSIZE": (
            float(ventes_taille.mean())
            if not ventes_taille.empty else 0.0
        ),
        "SALES_MOYEN_PAYS": (
            float(ventes_pays.mean())
            if not ventes_pays.empty else 0.0
        ),
    }

# ======================================================
# 6. ROUTE D'ACCUEIL
# ======================================================

@app.get("/")
def home():
    return {
        "message": "API de prédiction des ventes opérationnelle",
        "version": app.version,
        "documentation": "/docs",
    }

# ======================================================
# 7. ROUTE DE PRÉDICTION
# ======================================================

@app.post("/predire")
def predire(data: DonneesCommande):
    try:
        date_commande = pd.Timestamp(data.ORDERDATE)

        # Construire les variables de base
        df = pd.DataFrame([{
            "PRODUCTLINE": data.PRODUCTLINE,
            "DEALSIZE": data.DEALSIZE,
            "COUNTRY": data.COUNTRY,
            "STATUS": data.STATUS,
            "MONTH": date_commande.month,
            "YEAR": date_commande.year,
        }])

        df = ajouter_features(df)

        # Calcul automatique des statistiques historiques
        stats = calculer_stats_commande(
            nom_client=data.CUSTOMERNAME,
            date_commande=date_commande,
            productline=data.PRODUCTLINE,
            dealsize=data.DEALSIZE,
            country=data.COUNTRY,
        )

        for colonne, valeur in stats.items():
            df[colonne] = valeur

        # Respecter exactement les colonnes du modèle
        df = df[FEATURE_COLS]

        prediction_log = pipeline.predict(df)
        prediction = float(np.expm1(prediction_log[0]))

        if not np.isfinite(prediction):
            raise ValueError("La prédiction obtenue n'est pas valide.")

        prediction = max(0.0, prediction)

        return {
            "prediction_ventes": round(prediction, 2),
            "unite": "dollars ($)",
            "date_commande": date_commande.date().isoformat(),
            "statistiques_historiques": stats,
        }

    except HTTPException:
        raise
    except Exception:
        logger.exception("Échec lors de la prédiction")
        raise HTTPException(
            status_code=500,
            detail="La prédiction a échoué. Vérifiez les données fournies.",
        )
