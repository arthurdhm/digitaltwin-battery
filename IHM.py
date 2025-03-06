import streamlit as st
import requests
import pandas as pd
import plotly.express as px

st.title("🔋 Tableau de Bord de Supervision d'une Batterie")
st.markdown(
    "Ce tableau de bord permet de visualiser et simuler l'état d'une batterie avec des prédictions de durée de vie et ses métadonnées."
)

# 🎛️ **Section API & Paramètres**
st.sidebar.header("⚙️ Paramètres de Simulation")

# 🔧 **Entrée utilisateur pour les paramètres de la batterie**
charge_cycles = st.sidebar.slider("Cycles de charge", 0, 250, 150, 10)
temperature = st.sidebar.slider("Température (°C)", -10, 80, 25)
voltage = st.sidebar.slider("Tension (V)", 1.7, 4.5, 3.7)
current = st.sidebar.slider("Courant (A)", 0.5, 2.5, 1.5, 0.1)
capacity = st.sidebar.slider("Capacité (Ah)", 0.0, 3.0, 2.5, 0.5)

# 📡 **Appel API RUL (Durée de Vie Restante)**
rul_api_url = "http://localhost:5000/predict"  # Remplace par l'URL réelle
data_rul = {
    "Voltage_measured": voltage,
    "Current_measured": current,
    "Temperature_measured": temperature,
    "Current_charge": current,  
    "Voltage_charge": voltage,
    "Capacity": capacity
}

try:
    response_rul = requests.post(rul_api_url, json=data_rul)
    if response_rul.status_code == 200:
        rul_prediction = response_rul.json().get("predicted_RUL", "N/A")
        st.sidebar.success(f"🔮 Durée de vie restante estimée: {rul_prediction} cycles")
    else:
        rul_prediction = "Erreur API"
        st.sidebar.error(f"⚠️ Erreur API ({response_rul.status_code}) : {response_rul.text}")
except Exception as e:
    rul_prediction = "Erreur API"
    st.sidebar.error(f"❌ Exception : {e}")

# 🌐 **Appel API AAS (Métadonnées de la Batterie)**
aas_api_url = "http://159.84.130.201:8081/submodels/aHR0cHM6Ly9leGFtcGxlLmNvbS9pZHMvc20vNTIxMF8wMTIyXzExNDJfNzg5OQ/submodel-elements?encodedCursor=string&decodedCursor=string&level=deep&extent=withoutBlobValue"  # Remplace par l'URL correcte
try:
    response_aas = requests.get(aas_api_url)
    if response_aas.status_code == 200:
        aas_metadata = response_aas.json()
    else:
        aas_metadata = {"Erreur": "API non accessible"}
        st.sidebar.warning("⚠️ Impossible de récupérer les métadonnées de la batterie.")
except Exception as e:
    aas_metadata = {"Erreur": str(e)}

# 📊 **Visualisation des paramètres de la batterie**
st.subheader("📈 État Actuel de la Batterie")
col1, col2, col3 = st.columns(3)
col1.metric("⚡ Cycles de Charge", f"{charge_cycles} cycles")
col2.metric("🌡️ Température", f"{temperature} °C")
col3.metric("🔋 Tension", f"{voltage} V")

col4, col5 = st.columns(2)
col4.metric("🔌 Courant", f"{current} A")
col5.metric("🛠️ Capacité", f"{capacity} Ah")

# 📊 **Graphique des paramètres**
data = pd.DataFrame({
    "Paramètres": ["Cycles de Charge", "Température (°C)", "Tension (V)", "Courant (A)", "Capacité (Ah)"],
    "Valeurs": [charge_cycles, temperature, voltage, current, capacity]
})
fig = px.bar(data, x="Paramètres", y="Valeurs", title="🔋 État Actuel de la Batterie", text="Valeurs")
st.plotly_chart(fig)

# 🔮 **Simulation de la durée de vie restante**
st.subheader("📉 Simulation de l'Évolution de la Batterie")
cycles = range(0, charge_cycles + 500, 50)
predicted_rul = [
    max(float(rul_prediction) - cycle * 0.05, 0) if rul_prediction not in ["Erreur API", "N/A"] else 0
    for cycle in cycles
]

simulation_data = pd.DataFrame({
    "Cycles": cycles,
    "Durée de vie restante (estimée)": predicted_rul
})
simulation_fig = px.line(
    simulation_data, x="Cycles", y="Durée de vie restante (estimée)",
    title="📉 Évolution Simulée de la Durée de Vie"
)
st.plotly_chart(simulation_fig)

# 🔎 **Affichage des métadonnées AAS**
st.subheader("📜 Métadonnées de la Batterie")
st.json(aas_metadata)

# ℹ️ **Instructions**
st.markdown("### 📌 Instructions")
st.write("- 🔧 Ajustez les sliders à gauche pour simuler des scénarios de batterie.")
st.write("- 🔄 Les données sont récupérées en temps réel depuis les APIs RUL et AAS.")
st.write("- 📊 Les nouvelles métriques de courant et de capacité améliorent la précision des prédictions.")
