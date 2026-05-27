# 📈 Prédiction des Ventes

## 🎯 Objectif
Prédire les ventes futures à partir des caractéristiques 
d'une commande sans connaître le prix ni la quantité.

## 📊 Dataset
- 2748 enregistrements
- Période : 2018 - 2020
- Cible : SALES

## 🤖 Modèle retenu
- **XGBoost + log1p + Feature Engineering**
- R² : 0.7778
- MAE : 746.85

## 🛠️ Technologies
- Python 3.11
- Pandas, NumPy, Matplotlib
- Scikit-learn, XGBoost
- FastAPI, Uvicorn
- Streamlit
- Joblib

## 🚀 Installation

### 1. Cloner le projet
git clone https://github.com/Coul2704/predict-sales.git
cd predict-sales

### 2. Installer les dépendances
pip install -r requirements.txt

### 3. Lancer l'API
cd api
uvicorn api:app --reload

### 4. Lancer le Dashboard
cd dashboard
streamlit run streamlit_app.py

## 👤 Auteur
Coulibaly Zana Daniel Wilfried