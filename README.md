# 📈 Predict_Sales — Prédiction des ventes

## Présentation

**Predict_Sales** est un projet de Machine Learning visant à prédire le montant des ventes d'une commande à partir de ses caractéristiques commerciales, temporelles et historiques.

Le projet comprend un pipeline de Machine Learning, une API REST avec FastAPI et un tableau de bord interactif avec Streamlit.

## Objectifs

- Analyser les données historiques de ventes.
- Étudier les tendances commerciales et temporelles.
- Construire un modèle prédictif à l'aide de XGBoost.
- Exposer le modèle via une API REST.
- Proposer une interface pour effectuer des prédictions.

## Jeu de données

- **Nombre d'enregistrements :** 2 748
- **Période étudiée :** 2018–2020
- **Variable cible :** `SALES`

Les variables exploitées comprennent notamment la gamme de produits, la taille de la commande, le pays, le statut et des indicateurs historiques calculés à partir des ventes antérieures.

## Modèle retenu

Le modèle repose sur **XGBoost**, intégré dans un pipeline comprenant le prétraitement des variables catégorielles et une transformation logarithmique de la cible.

L'évaluation utilise une séparation temporelle : les données antérieures à 2020 servent à l'entraînement, tandis que les données de 2020 sont utilisées pour le test.

| Indicateur | Résultat |
|---|---:|
| Données d'entraînement | 2 282 |
| Données de test | 465 |
| R² | 0,7794 |
| MAE | 748,58 |

Le R² indique la capacité du modèle à expliquer la variabilité des ventes. La MAE représente l'erreur absolue moyenne, exprimée dans l'unité de la variable `SALES`.

## Technologies

- Python
- Pandas et NumPy
- Scikit-learn
- XGBoost et Joblib
- FastAPI et Uvicorn
- Streamlit et Requests
- Matplotlib
- Jupyter Notebook

## Structure du projet

```text
Predict_Sales/
├── api/
│   └── api.py
├── dashboard/
│   └── streamlit_app.py
├── data/
│   └── Auto Sales data.csv
├── models/
│   └── sales_pipeline.pkl
├── notebooks/
│   ├── Analyse descriptive.ipynb
│   ├── Analyse client.ipynb
│   ├── Analyse commerciale.ipynb
│   ├── Analyse des statuts de commande.ipynb
│   ├── Analyse produit.ipynb
│   ├── Analyse temporelle.ipynb
│   └── Modélisation.ipynb
├── .gitignore
└── requirements.txt
```

## Installation

### 1. Cloner le dépôt

```bash
git clone https://github.com/Coul2704/Predict_Sales.git
cd Predict_Sales
```

### 2. Créer un environnement virtuel

Sous Windows :

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Installer les dépendances

```bash
python -m pip install -r requirements.txt
```

### 4. Lancer l'API

Depuis la racine du projet :

```bash
python -m uvicorn api.api:app --reload
```

L'API sera normalement accessible à l'adresse `http://127.0.0.1:8000`. La documentation interactive FastAPI sera disponible sur `http://127.0.0.1:8000/docs`.

### 5. Lancer le dashboard

Dans un second terminal, active le même environnement virtuel, puis exécute :

```bash
python -m streamlit run dashboard/streamlit_app.py
```

Le dashboard doit communiquer avec l'API en cours d'exécution.

## Auteur

**Coulibaly Zana Daniel Wilfried**

Projet de Data Science et Machine Learning.
