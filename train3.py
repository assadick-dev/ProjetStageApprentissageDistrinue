import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

# --- 1. Lecture et nettoyage des données ---
df = pd.read_csv("data/raw/dataset.txt", sep=';')
df["Global_active_power"] = df["Global_active_power"].replace('?', np.nan)
df["Global_active_power"] = df["Global_active_power"].astype(np.float32)
df = df.dropna(subset=["Global_active_power"]).reset_index(drop=True)

# --- 2. Normalisation ---
scaler = MinMaxScaler()
serie = df["Global_active_power"].values.reshape(-1, 1)
serie_scaled = scaler.fit_transform(serie).flatten()
joblib.dump(scaler, "scaler.save")

# --- 3. Création des séquences ---
WINDOW_SIZE = 24
EPOCHS = 10
BATCH_SIZE = 32

def make_sequences(arr, window_size):
    X, y = [], []
    for i in range(len(arr) - window_size):
        X.append(arr[i:i+window_size])
        y.append(arr[i+window_size])
    return np.array(X), np.array(y)

X_train, y_train = make_sequences(serie_scaled, WINDOW_SIZE)
print("Shape X_train :", X_train.shape)
print("Shape y_train :", y_train.shape)

# --- 4. Récupération TF_CONFIG & MultiWorkerMirroredStrategy (AVANT toute op TF) ---
import tensorflow as tf
tf_config = json.loads(os.environ.get('TF_CONFIG', '{"cluster": {"worker": ["localhost:12345"]}, "task": {"type": "worker", "index": 0}}'))
num_workers = len(tf_config['cluster']['worker'])
strategy = tf.distribute.MultiWorkerMirroredStrategy()

# --- 5. Dataset TensorFlow (APRES la stratégie!) ---
with strategy.scope():
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_ds = train_ds.shuffle(buffer_size=1024).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    model = tf.keras.Sequential([
        tf.keras.Input(shape=(WINDOW_SIZE,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(0.001), metrics=["mae"])

def is_chief():
    task = tf_config.get("task", {})
    return task.get("type") == "worker" and task.get("index") == 0

callbacks = []
if is_chief():
    os.makedirs("models", exist_ok=True)
    callbacks.append(tf.keras.callbacks.ModelCheckpoint(
        "models/best_model.h5", save_best_only=True, monitor="loss", verbose=1
    ))

# --- 6. Entraînement ---
model.fit(train_ds, epochs=EPOCHS, callbacks=callbacks)

if is_chief():
    model.save("models/final_model.h5")
    print("Modèle final sauvegardé dans models/final_model.h5")


