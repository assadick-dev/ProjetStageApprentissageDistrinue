# utils/data_loader.py

import os
import tensorflow as tf

def parse_global_active(line):
    """
    Extrait la valeur Global_active_power (colonne 2) depuis une ligne
    Date;Time;Global_active_power;... et retourne un float32.
    Version robuste qui vérifie la présence des colonnes.
    """
    parts = tf.strings.split(line, sep=';')
    return tf.cond(
        tf.size(parts) > 2,
        lambda: tf.strings.to_number(parts[2], out_type=tf.float32),
        lambda: tf.constant(float('nan'), dtype=tf.float32)
    )

def build_datasets(batch_size=64,
                   window_size=24,
                   forecast_horizon=20,
                   shuffle_buffer=10000):
    """Version robuste du chargement des données avec vérifications supplémentaires"""
    
    # 1) Trouve dataset.txt avec vérification plus stricte
    cwd = os.getcwd()
    possible_paths = [
        os.path.join(cwd, "data", "raw", "dataset.txt"),
        os.path.join(cwd, "data", "dataset.txt"),
        os.path.join(cwd, "dataset.txt")
    ]
    
    txt_path = None
    for path in possible_paths:
        if tf.io.gfile.exists(path):
            txt_path = path
            break
    
    if txt_path is None:
        raise FileNotFoundError(f"Dataset introuvable. Chemins testés: {possible_paths}")

    # 2) Lecture avec filtres améliorés
    ds = (tf.data.TextLineDataset(txt_path)
          .skip(1)  # Saute l'en-tête
          .filter(lambda line: 
              tf.logical_and(
                  tf.logical_not(tf.strings.regex_full_match(line, r'.*\?.*')),  # Pas de valeurs manquantes
                  tf.size(tf.strings.split(line, sep=';')) > 2  # Au moins 3 colonnes
              ))
          .map(parse_global_active, num_parallel_calls=tf.data.AUTOTUNE)
          .filter(lambda x: ~tf.math.is_nan(x))  # Filtre les NaN
    )

    # 3) Fenêtres glissantes avec vérification de taille
    total_window = window_size + forecast_horizon
    ds = (ds.window(total_window, shift=1, drop_remainder=True)
          .flat_map(lambda w: w.batch(total_window))
          .filter(lambda x: tf.size(x) == total_window)  # Vérifie la taille
          .map(lambda seq: (seq[:window_size], seq[window_size:]),
               num_parallel_calls=tf.data.AUTOTUNE)
    )

    # 4) Configuration finale avec options
    ds = (ds
          .shuffle(shuffle_buffer)
          .batch(batch_size, drop_remainder=True)
          .prefetch(tf.data.AUTOTUNE)
    )
    
    # Désactive l'auto-sharding pour éviter des problèmes en distribution
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = (
        tf.data.experimental.AutoShardPolicy.OFF
    )
    ds = ds.with_options(options)
    
    return ds