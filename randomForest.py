import sys
import pandas as pd
import numpy as np
import sklearn
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor  # Importation du modèle Random Forest

# Charger les données
data = pd.read_csv("discharge.csv")

# Vérification des colonnes nécessaires dans les données
required_columns = ['Battery', 'Capacity', 'Voltage_measured', 'Current_measured', 'Temperature_measured', 'Current_charge', 'Voltage_charge', 'id_cycle']
missing_columns = [col for col in required_columns if col not in data.columns]
if missing_columns:
    print(f"Colonnes manquantes : {missing_columns}")
    sys.exit()
else:
    print("Toutes les colonnes nécessaires sont présentes.")

# Capacité nominale et tension nominale
C_nominal = 2.0  # Capacité nominale en Ah
V_nominal = 3.7  # Tension nominale en V

# Calcul de la capacité initiale
data['Capacity_initial'] = data.groupby('Battery', observed=False)['Capacity'].transform('max')

# Calcul de l'Index de Santé (HI)
data['HI'] = (data['Capacity'] / data['Capacity_initial']) * (data['Voltage_measured'] / V_nominal) * 100

# Déterminer le seuil de fin de vie (EOL)
EOL_threshold = 0.7 * data['Capacity_initial']

def calculate_cycle_max(group):
    capacity_threshold = 0.7 * group['Capacity_initial'].iloc[0]
    cycle_max = group[group['Capacity'] >= capacity_threshold]['id_cycle'].max()
    return cycle_max if not pd.isna(cycle_max) else group['id_cycle'].max()

data['Cycle_max'] = data.groupby('Battery', group_keys=False).apply(calculate_cycle_max)
data['Cycle_max'] = data['Cycle_max'].fillna(data.groupby('Battery')['id_cycle'].transform('max'))

data['RUL'] = data['Cycle_max'] - data['id_cycle']

# Séparer les batteries pour l'entraînement et le test
train_batteries = ['B0018', 'B0006', 'B0007']
test_battery = 'B0005'

train_data = data[data['Battery'].isin(train_batteries)]
test_data = data[data['Battery'] == test_battery]

X_train = train_data[["Voltage_measured", "Current_measured", "Temperature_measured", "Current_charge", "Voltage_charge", "Capacity"]]
y_train = train_data['RUL']

X_test = test_data[["Voltage_measured", "Current_measured", "Temperature_measured", "Current_charge", "Voltage_charge", "Capacity"]]
y_test = test_data['RUL']

# Mise à l'échelle des données
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Modèle Random Forest
model_rf = RandomForestRegressor(n_estimators=100, random_state=42)

# Entraînement du modèle
model_rf.fit(X_train_scaled, y_train)

# Prédictions
y_train_pred_rf = model_rf.predict(X_train_scaled)
y_test_pred_rf = model_rf.predict(X_test_scaled)

# Évaluation des performances
train_mse_rf = mean_squared_error(y_train, y_train_pred_rf)
train_r2_rf = r2_score(y_train, y_train_pred_rf)
test_mse_rf = mean_squared_error(y_test, y_test_pred_rf)
test_r2_rf = r2_score(y_test, y_test_pred_rf)

# Affichage des résultats
print(f"--- Random Forest ---")
print(f"Sur l'ensemble d'entraînement :")
print(f"MSE : {train_mse_rf}")
print(f"R² : {train_r2_rf}")
print("-" * 30)
print(f"Sur l'ensemble de test :")
print(f"MSE : {test_mse_rf}")
print(f"R² : {test_r2_rf}")