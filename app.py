import numpy as np
import joblib
import pandas as pd
from flask import Flask, request, jsonify

# Charger le modèle et le scaler préalablement sauvegardés
model = joblib.load('xgboost_model.pkl')
scaler = joblib.load('scaler.pkl')  # Charger le scaler sauvegardé

# Initialiser l'application Flask
app = Flask(__name__)

# Fonction de prétraitement des données d'entrée
def preprocess_input(data):
    try:
        # Assurez-vous que les données sont dans le bon format (DataFrame)
        input_data = pd.DataFrame([data])  # Convertir en DataFrame correctement
        
        # Vérification de la présence des colonnes requises
        required_columns = ["Voltage_measured", "Current_measured", "Temperature_measured", 
                            "Current_charge", "Voltage_charge", "Capacity"]
        
        if not all(col in input_data.columns for col in required_columns):
            raise ValueError(f"Colonnes manquantes dans les données reçues : {set(required_columns) - set(input_data.columns)}")
        
        # Normaliser les données (⚠️ Utilisation du scaler déjà ajusté)
        input_data_scaled = scaler.transform(input_data)  # Ne pas refit le scaler !

        return input_data_scaled
    
    except Exception as e:
        raise ValueError(f"Erreur lors du prétraitement des données : {str(e)}")

# Définir une route pour la page d'accueil
@app.route('/')
def home():
    return "ML Model API is running!"

# Définir une route pour la prédiction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Récupérer les données envoyées via le POST
        data = request.get_json(force=True)

        # Vérifier si toutes les données nécessaires sont présentes
        required_fields = ["Voltage_measured", "Current_measured", "Temperature_measured", 
                           "Current_charge", "Voltage_charge", "Capacity"]
        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Missing required data fields'}), 400

        # Prétraiter les données d'entrée
        processed_data = preprocess_input(data)

        # Faire la prédiction avec le modèle chargé
        prediction = model.predict(processed_data)

        # ✅ Convertir en float natif Python pour éviter l'erreur JSON
        prediction_value = float(prediction[0])

        # Renvoyer la prédiction sous forme JSON
        return jsonify({'predicted_RUL': prediction_value})

    except ValueError as ve:
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Lancer l'application Flask
if __name__ == '__main__':
    app.run(debug=True)
