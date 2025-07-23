import os, json
import tensorflow as tf
from utils.data_loader import build_datasets

# 1) Récupérer automatiquement le nombre de workers
tf_config = json.loads(os.environ['TF_CONFIG'])
num_workers = len(tf_config['cluster']['worker'])

# 2) GLOBAL_BATCH total souhaité (ex. 16 ou 12)
GLOBAL_BATCH = 12
PER_WORKER_BATCH = max(GLOBAL_BATCH // num_workers, 1)

# 3) Hyperparamètres
WINDOW_SIZE = 32   # réduisez à 8 ou 6 si nécessaire
EPOCHS      = 5

# 4) Strategy et dataset
strategy = tf.distribute.MultiWorkerMirroredStrategy()
train_ds = build_datasets(batch_size=PER_WORKER_BATCH,
                          window_size=WINDOW_SIZE)



with strategy.scope():
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(WINDOW_SIZE,)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(1),
    ])
    model.compile(loss="mse", optimizer="adam", metrics=["mae"])

# Callbacks sur le chief uniquement
def is_chief():
    task = tf_config.get("task", {})
    return task.get("type")=="worker" and task.get("index")==0

callbacks = []
if is_chief():
    os.makedirs("models", exist_ok=True)
    callbacks.append(tf.keras.callbacks.ModelCheckpoint(
        "models/best_model.h5", save_best_only=True, monitor="loss", verbose=1
    ))

# 5) Entraînement

print(train_ds)
#model.fit(train_ds, epochs=EPOCHS, callbacks=callbacks)

#if is_chief():
    #model.save("models/final_model.h5")
    #print("Modèle final sauvegardé dans models/final_model.h5")