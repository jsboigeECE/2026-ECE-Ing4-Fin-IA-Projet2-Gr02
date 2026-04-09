import os

import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
path = os.path.join(PROJECT_ROOT, 'dataset', 'creditcard.csv')
print('Fichier existe:', os.path.exists(path))
if not os.path.exists(path):
    raise FileNotFoundError(path)
df = pd.read_csv(path)
print('Taille (MB):', round(os.path.getsize(path)/(1024*1024),3))
print('Lignes:', len(df))
print('Colonnes:', len(df.columns))
print('Noms colonnes[:10]:', list(df.columns[:10]))
print('Total NA:', int(df.isna().sum().sum()))
nas = df.isna().sum()[df.isna().sum()>0]
print('NA par colonne:', nas.to_dict())
print('Dtypes:', df.dtypes.value_counts().to_dict())
