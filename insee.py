import pandas as pd

# 1) Lecture des trois fichiers
daily   = pd.read_csv('output.txt',                   sep=';', dtype={'code_INSEE': str})
monthly = pd.read_csv('data/raw/swi_herault_with_communes.csv',sep=';', dtype={'code_insee':  str})
climate = pd.read_csv('data/raw/herault_climate_data.csv',     sep=',', dtype={'LAMBX': float, 'LAMBY': float})

# 2) Propager le code_INSEE dans le fichier 'climate'
#    en se servant de la table 'monthly' qui associe chaque maille (LAMBX,LAMBY) à un code_insee
coord_code = (
    monthly[['LAMBX','LAMBY','code_insee']]
    .drop_duplicates()
)
climate = climate.merge(
    coord_code,
    on=['LAMBX','LAMBY'],
    how='left'
)

# 3) Fusion daily ↔ monthly uniquement sur le code
df_dm = daily.merge(
    monthly,
    left_on  = 'code_INSEE',
    right_on = 'code_insee',
    how       = 'outer',
    suffixes = ('','_mensuel')
)

# 4) Fusion df_dm ↔ climate uniquement sur le code
df_final = df_dm.merge(
    climate,
    left_on  = 'code_INSEE',
    right_on = 'code_insee',
    how       = 'outer'
)

# 5) Écrire le résultat
df_final.to_csv('fusion.txt', sep=';', index=False, encoding='utf-8')