import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Predicción Deportiva", layout="centered")
st.title("🔮 Predicción de Resultados Deportivos")

try:
    model = joblib.load("model.pkl")
except FileNotFoundError:
    st.error("El modelo 'model.pkl' no se encontró.")
    st.stop()

st.subheader("📋 Datos del Partido")
local_goles = st.number_input("Goles del equipo local", min_value=0, step=1)
visitante_goles = st.number_input("Goles del equipo visitante", min_value=0, step=1)
posesion_local = st.slider("Posesión del equipo local (%)", 0, 100, 50)
posesion_visitante = 100 - posesion_local

if st.button("Predecir resultado"):
    new_match = pd.DataFrame([{
        "local_goles": local_goles,
        "visitante_goles": visitante_goles,
        "posesion_local": posesion_local,
        "posesion_visitante": posesion_visitante
    }])
    probs = model.predict_proba(new_match)[0]
    st.success("✅ Predicción generada")
    st.metric("Victoria Local", f"{probs[0]:.2%}")
    st.metric("Empate", f"{probs[1]:.2%}")
    st.metric("Victoria Visitante", f"{probs[2]:.2%}")

    # Visualización de las probabilidades
    fig, ax = plt.subplots()
    etiquetas = ["Victoria Local", "Empate", "Victoria Visitante"]
    ax.bar(etiquetas, probs, color=["green", "orange", "red"])
    ax.set_ylabel("Probabilidad")
    ax.set_ylim(0, 1)
    st.pyplot(fig)