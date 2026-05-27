import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(
    page_title="Prédiction des Ventes",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Application Prédiction des Ventes")

st.markdown("""
Cette application utilise un modèle Machine Learning XGBoost
pour prédire le montant des ventes.
""")

# =========================================================
# API
# =========================================================
st.sidebar.header("⚙️ Configuration API")

API_URL = st.sidebar.text_input(
    "URL API FastAPI",
    value="http://127.0.0.1:8000/predire"
)

# =========================================================
# INPUTS
# =========================================================
st.subheader("📋 Données de la commande")

col1, col2 = st.columns(2)

with col1:

    productline = st.selectbox(
        "Product Line",
        ["Classic Cars","Motorcycles","Planes","Ships","Trains","Trucks and Buses","Vintage Cars"]
    )

    dealsize = st.selectbox(
        "Deal Size",
        ["Small","Medium","Large"]
    )

    country = st.selectbox(
        "Country",
         [
            "USA", "France", "Germany", "UK", "Australia",
            "Spain", "Italy", "Japan", "Canada", "Norway",
            "Belgium", "Finland", "Austria", "Denmark",
            "Switzerland", "Sweden", "Singapore",
            "Philippines", "Ireland"
        ]
    )

with col2:

    status = st.selectbox(
        "Status",
        ["Shipped","Cancelled","Disputed","In Process","On Hold","Resolved"]
    )

    month = st.slider("Month", 1, 12, 6)
    year = st.number_input("Year", 2020, 2035, 2021)

# =========================================================
# STATS (features du modèle)
# =========================================================
st.subheader("📊 Features statistiques")

col3, col4 = st.columns(2)

with col3:
    sales_moyen_client = st.number_input("Sales moyen client", 0.0, 1e9, 3500.0)
    sales_total_client = st.number_input("Sales total client", 0.0, 1e9, 35000.0)

with col4:
    sales_moyen_produit = st.number_input("Sales moyen produit", 0.0, 1e9, 2800.0)
    sales_moyen_dealsize = st.number_input("Sales moyen dealsize", 0.0, 1e9, 3000.0)
    sales_moyen_pays = st.number_input("Sales moyen pays", 0.0, 1e9, 2900.0)

# =========================================================
# BUTTON
# =========================================================
st.markdown("---")

if st.button("🔮 Prédire", use_container_width=True):

    payload = {
        "PRODUCTLINE": productline,
        "DEALSIZE": dealsize,
        "COUNTRY": country,
        "STATUS": status,
        "MONTH": month,
        "YEAR": int(year),

        "QUARTER": ((month - 1) // 3) + 1,
        "FIN_ANNEE": 1 if month in [10,11,12] else 0,
        "SAISON": "Hiver" if month in [12,1,2]
                  else "Printemps" if month in [3,4,5]
                  else "Ete" if month in [6,7,8]
                  else "Automne",

        "SALES_MOYEN_CLIENT": sales_moyen_client,
        "SALES_TOTAL_CLIENT": sales_total_client,
        "SALES_MOYEN_PRODUIT": sales_moyen_produit,
        "SALES_MOYEN_DEALSIZE": sales_moyen_dealsize,
        "SALES_MOYEN_PAYS": sales_moyen_pays
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=10)

        if response.status_code == 200:

            result = response.json()
            prediction = result["prediction_ventes"]

            st.success("✅ Prédiction réussie")

            st.metric("💰 Ventes prévues", f"{prediction:,.2f} $")

            # =================================================
            # GRAPH
            # =================================================
            df_visu = pd.DataFrame({
                "Indicateur": [
                    "Client",
                    "Produit",
                    "DealSize",
                    "Pays",
                    "Prediction"
                ],
                "Valeur": [
                    sales_moyen_client,
                    sales_moyen_produit,
                    sales_moyen_dealsize,
                    sales_moyen_pays,
                    prediction
                ]
            })

            fig, ax = plt.subplots()
            ax.bar(df_visu["Indicateur"], df_visu["Valeur"])
            ax.set_title("Comparaison")
            st.pyplot(fig)

            # =================================================
            # INTENSITE
            # =================================================
            if prediction < 1000:
                st.warning("Faible potentiel")
            elif prediction < 5000:
                st.info("Potentiel moyen")
            else:
                st.success("Fort potentiel")

            with st.expander("📦 Données envoyées"):
                st.json(payload)

        else:
            st.error(f"Erreur API {response.status_code}")
            st.json(response.text)

    except requests.exceptions.ConnectionError:
        st.error("❌ API non accessible")

    except Exception as e:
        st.error(str(e))