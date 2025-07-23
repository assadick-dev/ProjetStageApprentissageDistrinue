import tensorflow as tf
import numpy as np

# Charger le modèle entraîné
model = tf.keras.models.load_model("models/final_model.h5", compile=False)

# Tes 24 dernières valeurs de Global_active_power
last_window = [
    3.232, 3.254, 3.376, 3.372, 3.366, 3.386, 3.496, 3.426, 3.440, 3.440, 3.454, 3.450,
    3.480, 3.474, 3.456, 1.832, 2.044, 3.354, 3.282, 3.308, 3.314, 3.400, 3.400, 3.388
]

# Mise en forme (1, 24)
current_window = np.array(last_window, dtype=np.float32).reshape(1, -1)

# Génère 20 prédictions en rolling
predictions = []
for _ in range(20):
    pred = model.predict(current_window, verbose=0)[0, 0]
    print(pred)
    predictions.append(pred)
    # Rolling : décale, ajoute la prédiction à la fin
    current_window = np.roll(current_window, -1)
    current_window[0, -1] = pred

print("\nLes 20 prochaines prédictions (kW) :")
for i, val in enumerate(predictions, 1):
    print(f"{i}: {val:.3f} kW")