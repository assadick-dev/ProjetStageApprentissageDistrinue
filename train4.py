import os
import json
import tensorflow as tf
from utils.data_loader import build_datasets
import numpy as np
import matplotlib.pyplot as plt

# Configuration
tf_config = json.loads(os.environ.get('TF_CONFIG', '{}'))
num_workers = len(tf_config.get('cluster', {}).get('worker', [1]))

GLOBAL_BATCH = 32
PER_WORKER_BATCH = max(GLOBAL_BATCH // num_workers, 1)
WINDOW_SIZE = 24
EPOCHS = 50  # Augmenté pour permettre la convergence
FORECAST_HORIZON = 5
LEARNING_RATE = 0.0001  # Réduit pour une meilleure stabilité

# Stratégie de distribution
strategy = tf.distribute.MultiWorkerMirroredStrategy()

# Chargement des données
train_ds, _ = build_datasets(
    batch_size=PER_WORKER_BATCH,
    window_size=WINDOW_SIZE,
    forecast_horizon=FORECAST_HORIZON
)

# Extraction des données pour normalisation
def extract_features(dataset):
    features = []
    for batch in dataset:
        features.append(batch[0].numpy())
    return np.concatenate(features, axis=0)

X_all = extract_features(train_ds)

# Vérification des données
print("\n=== Statistiques des données ===")
print(f"Shape: {X_all.shape}")
print(f"Min: {np.min(X_all):.2f}, Max: {np.max(X_all):.2f}")
print(f"Mean: {np.mean(X_all):.2f}, Std: {np.std(X_all):.2f}")

# Construction du modèle
with strategy.scope():
    # Couche de normalisation
    norm_layer = tf.keras.layers.Normalization()
    norm_layer.adapt(X_all)
    
    # Architecture améliorée
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(WINDOW_SIZE,)),
        norm_layer,
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(FORECAST_HORIZON)
    ])
    
    # Compilation avec métriques supplémentaires
    model.compile(
        loss="mse",
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
        metrics=['mae', 'mape']
    )

# Callbacks
def is_chief():
    task = tf_config.get("task", {})
    return task.get("type") == "worker" and task.get("index") == 0

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='loss',
        patience=10,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6
    )
]

if is_chief():
    os.makedirs("models", exist_ok=True)
    callbacks.append(tf.keras.callbacks.ModelCheckpoint(
        "models/best_model.h5",
        save_best_only=True,
        monitor="loss",
        verbose=1
    ))
    callbacks.append(tf.keras.callbacks.CSVLogger(
        "models/training_log.csv"
    ))

# Inspection d'un batch
print("\n=== Exemple de batch ===")
for batch in train_ds.take(1):
    X_batch, y_batch = batch
    print("Input shape:", X_batch.shape)
    print("Target shape:", y_batch.shape)
    print("Sample input:", X_batch[0].numpy())
    print("Sample target:", y_batch[0].numpy())

# Entraînement
print("\n=== Début de l'entraînement ===")
history = model.fit(
    train_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

# Sauvegarde finale
if is_chief():
    model.save("models/final_model.h5")
    print("\nModèle final sauvegardé dans models/final_model.h5")
    
    # Visualisation des métriques
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.title('Evolution de la loss pendant l\'entraînement')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.savefig("models/training_curve.png")
    plt.close()

# Validation visuelle
sample_X, sample_y = next(iter(train_ds))
preds = model.predict(sample_X)

print("\n=== Validation ===")
print("Vraies valeurs (5 premiers pas):", sample_y.numpy()[0, :5])
print("Prédictions (5 premiers pas):", preds[0, :5])

# Calcul des erreurs
errors = sample_y.numpy() - preds
print("\nErreurs (moyenne/std):", np.mean(errors), np.std(errors))