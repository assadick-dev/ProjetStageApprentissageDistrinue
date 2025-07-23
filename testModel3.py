# test_active_compare_clean.py

import numpy as np
import tensorflow as tf
import pandas as pd

# ── CONFIG ───────────────────────────────────────────────────────────────────────
MODEL_PATH       = "models/final_model.h5"
DATA_PATH        = "data/raw/test.txt"   # ton fichier complet
WINDOW_SIZE      = 24
FORECAST_HORIZON = 1

# ── CHARGEMENT DES DONNÉES ──────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH, sep=";", decimal=".")

series = df["Global_active_power"].astype(np.float32).values

# ── CHARGEMENT DU MODÈLE ─────────────────────────────────────────────────────────
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# ── BALAYAGE & COMPARAISON ───────────────────────────────────────────────────────
records = []
for start in range(len(series) - WINDOW_SIZE - FORECAST_HORIZON + 1):
    window   = series[start : start + WINDOW_SIZE]
    true_val = series[start + WINDOW_SIZE]
    X_input  = window.reshape(1, WINDOW_SIZE)
    pred_val = float(model.predict(X_input, verbose=0)[0, 0])
    records.append({"true_1": true_val, "pred_1": pred_val})

df_compare = pd.DataFrame(records)

# Affichage
print(df_compare.head(20))
print("…")
print(df_compare.tail(20))

# Optionnel : tracé
import matplotlib.pyplot as plt
plt.plot(df_compare["true_1"], label="Vraie")
plt.plot(df_compare["pred_1"], label="Prédite", alpha=0.7)
plt.legend()
plt.title("Comparaison vraie vs prédite (pas suivant)")
plt.show()