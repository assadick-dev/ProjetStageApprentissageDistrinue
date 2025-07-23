import os, json
import tensorflow as tf
from utils.data_loader import build_datasets
import numpy as np

tf_config = json.loads(os.environ['TF_CONFIG'])
num_workers = len(tf_config['cluster']['worker'])

GLOBAL_BATCH = 64
PER_WORKER_BATCH = max(GLOBAL_BATCH // num_workers, 1)
WINDOW_SIZE = 24
EPOCHS = 50
FORECAST_HORIZON = 1  # Prédire 20 pas à l'avance

strategy = tf.distribute.MultiWorkerMirroredStrategy()
train_ds = build_datasets(
    batch_size=PER_WORKER_BATCH,
    window_size=WINDOW_SIZE,
    forecast_horizon=FORECAST_HORIZON
)

# Extrait X_train pour la normalisation
X_all = []
for batch in train_ds:
    X_all.append(batch[0].numpy())
X_all = np.concatenate(X_all, axis=0)





with strategy.scope():
    norm_layer = tf.keras.layers.Normalization()
    norm_layer.adapt(X_all)

    model = tf.keras.Sequential([
        tf.keras.Input(shape=(WINDOW_SIZE,)),
        norm_layer,
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)  # <= Multi-output
    ])
    model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(0.001))

def is_chief():
    task = tf_config.get("task", {})
    return task.get("type") == "worker" and task.get("index") == 0

callbacks = []
if is_chief():
    os.makedirs("models", exist_ok=True)
    callbacks.append(tf.keras.callbacks.ModelCheckpoint(
        "models/best_model.h5", save_best_only=True, monitor="loss", verbose=1
    ))

# Exemple affichage d'un batch pour contrôle
#for batch in train_ds.take(1):
   # X_batch, y_batch = batch
   # print("Exemple d'entrée X_batch :", X_batch.numpy())
    #print("Exemple de sortie y_batch :", y_batch.numpy())
    #print("Shape X_batch :", X_batch.shape)
   #print("Shape y_batch :", y_batch.shape)

model.fit(train_ds, epochs=EPOCHS, callbacks=callbacks)

if is_chief():
    model.save("models/final_model.h5")
    print("Modèle final sauvegardé dans models/final_model.h5")