import streamlit as st
import joblib
import pandas as pd
import datetime

# ✅ Charger le bon modèle et les colonnes
model = joblib.load("xgb_home_credit_pipeline_streamlit.joblib")
numeric_cols, categorical_cols = joblib.load("features_used_streamlit.joblib")
all_cols = numeric_cols + categorical_cols

# Configuration de la page
st.set_page_config(page_title="Scoring client - Néo Banque", layout="centered")
st.title("📊 Évaluation du risque client")

# 📝 Formulaire utilisateur
with st.form("formulaire"):
    nom = st.text_input("Nom", value="Dupont")
    prenom = st.text_input("Prénom", value="Jean")
    naissance = st.date_input("Date de naissance", value=datetime.date(1980, 1, 1))
    statut = st.text_input("Statut", value="Cadre")  # Placeholder, non utilisé dans le modèle
    revenus = st.number_input("Revenus (€)", value=4500)
    credit = st.number_input("Montant du crédit demandé (€)", value=100000)
    adresse = st.text_input("Adresse", value="123 Rue Exemple")  # Placeholder aussi
    submit = st.form_submit_button("Évaluer le client")

# ⚙️ Prédiction si formulaire soumis
if submit:
    # Construction d'un dictionnaire avec valeurs par défaut
    input_data = {col: 0.0 for col in all_cols}

    # Remplir avec les vraies valeurs du formulaire
    input_data["AMT_INCOME_TOTAL"] = revenus
    input_data["AMT_CREDIT"] = credit
    # Remplacer GENDER et CONTRACT_TYPE par valeurs par défaut ou ajout de sélection si souhaité
    input_data["GENDER"] = 1  # 1 = M, 0 = F → à personnaliser si tu veux
    input_data["NAME_CONTRACT_TYPE"] = 0  # ex. 0 = Cash loans (à adapter)

    # Conversion âge → DAYS_BIRTH si utilisé (pas utilisé dans ton modèle actuel)
    # input_data["DAYS_BIRTH"] = -((pd.Timestamp.today() - pd.to_datetime(naissance)).days)

    # Prédiction
    df = pd.DataFrame([input_data])
    proba = model.predict_proba(df)[0][1]
    decision = (
        "✅ Éligible" if proba < 0.4 else
        "⚠️ Risque modéré" if proba < 0.7 else
        "❌ Risque élevé"
    )

    # Affichage des résultats
    st.metric("Score prédictif", f"{proba:.2%}")
    st.success(f"Décision : {decision}")
