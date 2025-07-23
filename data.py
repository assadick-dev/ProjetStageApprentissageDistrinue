import numpy as np
import pandas as pd
import tensorflow as tf

MODEL_PATH = "models/final_model.h5"
TEST_DATA_PATH = "data/raw/test.txt"
WINDOW_SIZE = 8  # comme à l'entraînement

# 1. Charger les données
df = pd.read_csv(TEST_DATA_PATH, sep=';')

# 2. Prendre la colonne Global_active_power uniquement, convertir en float
series = df['Global_active_power'].astype(np.float32).values

# 3. Créer les séquences d'entrée (X) et les labels (y), comme dans build_datasets
def make_sequences(series, window_size):
    X, y = [], []
    for i in range(len(series) - window_size):
        X.append(series[i:i + window_size])
        y.append(series[i + window_size])
    return np.array(X), np.array(y)

X_test, y_test = make_sequences(series, WINDOW_SIZE)

print(f"Shape X_test : {X_test.shape}")  # (n_samples, 8)

# 4. Charger le modèle
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# 5. Prédire
preds = model.predict(X_test)

# 6. Afficher les résultats
for i in range(len(preds)):
    print(f"Input: {X_test[i]}, Prediction: {preds[i][0]:.3f}, Real: {y_test[i]:.3f}")