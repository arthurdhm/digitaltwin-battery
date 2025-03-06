# Prédiction de la Durée de Vie Restante (RUL) d'une Batterie

Ce projet utilise l'apprentissage automatique pour prédire la durée de vie restante (RUL) d'une batterie à partir de diverses mesures, telles que la capacité, la tension, le courant, et la température. Nous avons utilisé le modèle XGBoost pour effectuer les prédictions.

## Étapes du Projet

### 1. Préparation des Données
Le projet commence par le chargement et la préparation des données à partir du fichier `discharge.csv`, qui contient les mesures de performance de différentes batteries.

- **Colonnes utilisées** : 
  - `Battery`, `Capacity`, `Voltage_measured`, `Current_measured`, `Temperature_measured`, `Current_charge`, `Voltage_charge`, `id_cycle`
- Les colonnes sont vérifiées pour s'assurer que toutes les données nécessaires sont présentes.

### 2. Calcul de la Capacité Initiale et de l'Index de Santé (HI)
- La **Capacité Initiale** est calculée en prenant la valeur maximale de la capacité mesurée pour chaque batterie.
- Un **Index de Santé** (HI) est calculé en fonction de la capacité et de la tension mesurée de chaque batterie.

### 3. Détermination du Seuil de Fin de Vie (EOL)
- Un seuil de fin de vie (EOL) est défini comme 70% de la capacité initiale.
- Le **Cycle Max** est déterminé pour chaque batterie en identifiant le cycle maximum avant que la capacité de la batterie atteigne le seuil EOL.

### 4. Calcul de la Durée de Vie Restante (RUL)
La durée de vie restante (RUL) de chaque batterie est calculée comme la différence entre le cycle actuel et le cycle maximum.

### 5. Séparation des Données en Ensemble d'Entraînement et de Test
Les données sont divisées en ensembles d'entraînement et de test. Un sous-ensemble de batteries est utilisé pour entraîner le modèle, tandis qu'une batterie distincte est utilisée pour tester sa performance.

### 6. Mise à l'Échelle des Données
Les données sont mises à l'échelle à l'aide d'un `StandardScaler`, afin de garantir que les différentes caractéristiques (voltage, courant, etc.) ont la même échelle et contribuent de manière égale à l'entraînement du modèle.

### 7. Entraînement du Modèle XGBoost
Le modèle **XGBoost** est utilisé pour entraîner un modèle de régression sur les données mises à l'échelle, afin de prédire la durée de vie restante (RUL). Plusieurs paramètres sont ajustés pour éviter l'overfitting, tels que le taux d'apprentissage, le nombre d'arbres et la profondeur des arbres.

### 8. Évaluation du Modèle
Le modèle est évalué en utilisant des métriques telles que l'**Erreur Quadratique Moyenne (MSE)** et le **R²**, aussi bien sur l'ensemble d'entraînement que sur l'ensemble de test.

### 9. Résultats
Les résultats de l'évaluation du modèle (MSE et R²) sont affichés pour permettre une évaluation de la performance du modèle.

## Fichiers du Projet

- `discharge.csv`: Fichier de données contenant les mesures des batteries.
- `xgboost_model.py`: Script Python qui implémente l'entraînement et l'évaluation du modèle XGBoost.
- `scaler.pkl`: Fichier contenant l'objet `StandardScaler` utilisé pour mettre à l'échelle les données.
- `xgboost_model.pkl`: Fichier contenant le modèle XGBoost entraîné.
- `README.md`: Ce fichier avec la description du projet.

## Installation

Pour utiliser ce projet, vous devez d'abord créer un environnement virtuel et installer les dépendances listées dans `requirements.txt` :

1. Clonez ce dépôt :
   ```bash
   git clone https://github.com/arthurdhm/digitaltwin-battery.git

2. Créez un environnement virtuel :
   ```bash
   python -m venv venv
   
3. Activer env virtuel :
   ```bash
   .\venv\Scripts\activate
   
4. Install dépendances
   ```bash
   pip install -r requirements.txt

5. Launch API App 
   ```bash
   python .\app.py

6. Launch Steamlit App in new terminal 
   ```bash
   python -m streamlit run .\IHM.py