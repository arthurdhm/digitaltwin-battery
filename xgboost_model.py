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
from xgboost import XGBRegressor

# Charger les données
data = pd.read_csv("discharge.csv")

# Vérification des colonnes nécessaires dans les données
required_columns = ['Battery', 'Capacity', 'Voltage_measured', 'Current_measured', 'Temperature_measured', 'Current_charge', 'Voltage_charge', 'id_cycle']
missing_columns = [col for col in required_columns if col not in data.columns]
if missing_columns:
    print(f"Colonnes manquantes : {missing_columns}")
else:
    print("Toutes les colonnes nécessaires sont présentes.")

# Capacité nominale et tension nominale (pour information, mais non utilisées directement)
C_nominal = 2.0  # Capacité nominale en Ah
V_nominal = 3.7  # Tension nominale en V

# Calcul de la capacité initiale pour chaque batterie (max de la capacité mesurée)
data['Capacity_initial'] = data.groupby('Battery', observed=False)['Capacity'].transform('max')

# Calcul de l'Index de Santé (HI) en utilisant la capacité mesurée et la tension mesurée
data['HI'] = (data['Capacity'] / data['Capacity_initial']) * (data['Voltage_measured'] / V_nominal) * 100

# Déterminer le seuil de fin de vie (EOL) à 70% de la capacité initiale
EOL_threshold = 0.7 * data['Capacity_initial']

# Fonction pour déterminer le cycle max
def calculate_cycle_max(group):
    capacity_threshold = 0.7 * group['Capacity_initial'].iloc[0]
    cycle_max = group[group['Capacity'] >= capacity_threshold]['id_cycle'].max()
    
    if pd.isna(cycle_max):
        cycle_max = group['id_cycle'].max()
    return cycle_max

# Appliquer la fonction pour calculer Cycle_max tout en excluant les colonnes de regroupement
data['Cycle_max'] = data.groupby('Battery', group_keys=False).apply(calculate_cycle_max)

# Vérification des valeurs NaN dans Cycle_max et remplir les NaN avec la fin du cycle actuel
data['Cycle_max'] = data['Cycle_max'].fillna(data.groupby('Battery')['id_cycle'].transform('max'))

# Calculer le RUL : cycles restants avant d'atteindre le seuil EOL
data['RUL'] = data['Cycle_max'] - data['id_cycle']

# Affichage des premières lignes après ajout du RUL
print(data[['Battery', 'id_cycle', 'Capacity', 'Voltage_measured', 'HI', 'RUL']].head())

# Filtrer les données en fonction des batteries spécifiques pour l'entraînement et le test
train_batteries = ['B0018', 'B0006', 'B0007']
test_battery = 'B0005'

# Séparer les données en fonction des batteries d'entraînement et de test
train_data = data[data['Battery'].isin(train_batteries)]
test_data = data[data['Battery'] == test_battery]

# Sélectionner les caractéristiques d'entrée (X) et la variable cible (y)
X_train = train_data[["Voltage_measured", "Current_measured", "Temperature_measured", "Current_charge", "Voltage_charge", "Capacity"]]
y_train = train_data['RUL']

X_test = test_data[["Voltage_measured", "Current_measured", "Temperature_measured", "Current_charge", "Voltage_charge", "Capacity"]]
y_test = test_data['RUL']

# Mise à l'échelle des données (important pour certains modèles, comme XGBoost)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Modèle XGBoost avec ajustements pour éviter l'overfitting
model_xgb = XGBRegressor(
    n_estimators=200,        # Plus d'arbres, mais un taux d'apprentissage plus faible
    learning_rate=0.05,      # Taux d'apprentissage plus faible
    max_depth=6,             # Limiter la profondeur des arbres
    subsample=0.8,           # Utilisation d'un sous-échantillon des données pour chaque arbre
    colsample_bytree=0.8,    # Utilisation de moins de caractéristiques par arbre
    alpha=0.1,               # Régularisation L1 (Lasso)
    lambda_=0.1,             # Régularisation L2 (Ridge)
    random_state=42
)

# Entraînement du modèle
model_xgb.fit(X_train_scaled, y_train)

# Prédictions sur l'ensemble d'entraînement
y_train_pred_xgb = model_xgb.predict(X_train_scaled)

# Calcul du MSE et R² sur l'ensemble d'entraînement
train_mse_xgb = mean_squared_error(y_train, y_train_pred_xgb)
train_r2_xgb = r2_score(y_train, y_train_pred_xgb)

# Prédictions sur l'ensemble de test
y_test_pred_xgb = model_xgb.predict(X_test_scaled)

# Calcul du MSE et R² sur l'ensemble de test
test_mse_xgb = mean_squared_error(y_test, y_test_pred_xgb)
test_r2_xgb = r2_score(y_test, y_test_pred_xgb)

# Affichage des résultats d'entraînement et de test
print(f"--- XGBoost avec régularisation et ajustements ---")
print(f"Sur l'ensemble d'entraînement :")
print(f"MSE : {train_mse_xgb}")
print(f"R² : {train_r2_xgb}")
print("-" * 30)
print(f"Sur l'ensemble de test :")
print(f"MSE : {test_mse_xgb}")
print(f"R² : {test_r2_xgb}")
