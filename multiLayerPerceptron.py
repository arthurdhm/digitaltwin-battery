import sys
import pandas as pd
import numpy as np
import joblib  
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Charger les données
data = pd.read_csv("discharge.csv")

# Vérification des colonnes nécessaires
required_columns = ['Battery', 'Capacity', 'Voltage_measured', 'Current_measured', 'Temperature_measured', 'Current_charge', 'Voltage_charge', 'id_cycle']
missing_columns = [col for col in required_columns if col not in data.columns]
if missing_columns:
    raise ValueError(f"Colonnes manquantes : {missing_columns}")
print("✅ Toutes les colonnes nécessaires sont présentes.")

# Calcul des valeurs essentielles
C_nominal = 2.0  
V_nominal = 3.7  

data['Capacity_initial'] = data.groupby('Battery', observed=False)['Capacity'].transform('max')
data['HI'] = (data['Capacity'] / data['Capacity_initial']) * (data['Voltage_measured'] / V_nominal) * 100

# Calcul du cycle max
def calculate_cycle_max(group):
    capacity_threshold = 0.7 * group['Capacity_initial'].iloc[0]
    cycle_max = group[group['Capacity'] >= capacity_threshold]['id_cycle'].max()
    return cycle_max if not pd.isna(cycle_max) else group['id_cycle'].max()

data['Cycle_max'] = data.groupby('Battery', group_keys=False).apply(calculate_cycle_max)
data['Cycle_max'].fillna(data.groupby('Battery')['id_cycle'].transform('max'), inplace=True)
data['RUL'] = data['Cycle_max'] - data['id_cycle']

# Sélection des batteries pour train/test
train_batteries = ['B0018', 'B0006', 'B0007']
test_battery = 'B0005'

train_data = data[data['Battery'].isin(train_batteries)]
test_data = data[data['Battery'] == test_battery]

# Sélection des features et de la variable cible
features = ["Voltage_measured", "Current_measured", "Temperature_measured", "Current_charge", "Voltage_charge", "Capacity"]
X_train, y_train = train_data[features], train_data['RUL']
X_test, y_test = test_data[features], test_data['RUL']

# Standardisation des données
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Sauvegarde du scaler
joblib.dump(scaler, "scaler-MVP.pkl")

# Définition du modèle MLP
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(1, activation='linear')  # Sortie linéaire pour la régression
])

# Compilation du modèle
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

# Entraînement du modèle
history = model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, validation_data=(X_test_scaled, y_test), verbose=1)

# Sauvegarde du modèle
model.save("MVP_MLP.h5")

# Prédictions
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

# Évaluation
train_mse = mean_squared_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f"\n--- 🎯 Modèle MLP ---")
print(f"📊 Sur l'ensemble d'entraînement : MSE={train_mse:.4f}, R²={train_r2:.4f}")
print(f"📊 Sur l'ensemble de test : MSE={test_mse:.4f}, R²={test_r2:.4f}")

# Vérification du chargement du modèle et test
loaded_model = keras.models.load_model("MVP_MLP.h5")
loaded_scaler = joblib.load("scaler-MVP.pkl")

# Test de prédiction avec un exemple réel
sample_input = np.array(X_test.iloc[0]).reshape(1, -1)
sample_input_scaled = loaded_scaler.transform(sample_input)
sample_prediction = loaded_model.predict(sample_input_scaled)

print(f"\n🔮 Prédiction RUL pour un exemple de test : {sample_prediction[0][0]:.2f} cycles restants")
