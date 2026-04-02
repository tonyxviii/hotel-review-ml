import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

# cARICAMNETO DEI MODELLI 

@st.cache_resource
def load_models():
    model_dept = joblib.load("models/model_department.pkl")
    model_sent = joblib.load("models/model_sentiment.pkl")
    return model_dept, model_sent

model_dept, model_sent = load_models()

# 2. CONFIGURAZIONE PAGINA

st.set_page_config(
    page_title="Hotel Review Classifier",
    page_icon="🏨",
    layout="wide"
)

st.title("🏨 Hotel Review Classifier")
st.markdown("Analizza le recensioni hotel: scopri il reaparto e il sentiment.")

# 3. ANALISI SINGOLA RECENSIONE 

st.header("📝 Analizza una recensione")

title_input = st.text_input("Titolo della recensione:")
body_input = st.text_area("Testo della recensione:")

if st.button("Analizza"):
    if title_input.strip() == "" or body_input.strip() == "":
        st.warning("Inserisci sia il titolo che il testo.")
    else: 
        full_text = title_input + " " + body_input

        dept_pred = model_dept.predict([full_text])[0]
        sent_pred = model_sent.predict([full_text])[0]

        dept_proba = model_dept.predict_proba([full_text])[0]
        sent_proba = model_sent.predict_proba([full_text])[0]

        dept_confidence = max(dept_proba) * 100
        sent_confidence = max(sent_proba) * 100

        col1, col2 = st.columns(2)

        with col1:
            st.success(f"🏢 Reparto: **{dept_pred}**")
            st.write(f"Confidenza: {dept_confidence: .1f}%")

        with col2:
            if sent_pred == "positive":
                st.success(f"Sentiment: **{sent_pred}**")
            else:
                st.error(f"Sentiment: **{sent_pred}**")
            st.write(f"Confidenza: {sent_confidence: .1f}%")

# 4 ANALISI IN BATCH (UPLOAD CSV)

st.header("📂 Analizza un file CSV")

uploaded_file = st.file_uploader("Carica un CSV con colonne 'title e 'body':", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    if "title" not in df.columns or "body" not in df.columns:
        st.error("Il CSV deve avere le colonne 'title' e 'body'.")
    else:
        df["full_text"] = df["title"] + " " + df["body"]
        df["pred_department"] = model_dept.predict(df["full_text"])
        df["pred_sentiment"] = model_sent.predict(df["full_text"])

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        df["timestamp"] = timestamp

        st.success(f"Analizzate {len(df)} recensioni.")
        st.dataframe(df[["title", "body", "pred_department", "pred_sentiment", "timestamp"]])

        csv_export = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label = "⬇️ Scarica risultati CSV",
            data = csv_export,
            file_name = f"risultati_{timestamp}.csv",
            mime = "text/csv"
        )


