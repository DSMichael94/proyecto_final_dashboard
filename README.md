# Proyecto Final - Dashboard (Metodos Estadisticos, DOE computacional, Regularizacion)

Este repo contiene:
- app.py: Dashboard en Dash con Welch t-test, bootstrap de diferencia de medias, regresion lineal con Ridge (alpha), regresion logistica con umbral y C.
- analysis.py: utilidades de modelado y metricas.
- dataset.py: genera data/dataset_estudiantes.csv (datos sinteticos).
- requirements.txt, Procfile, runtime.txt: correr local y preparar Binder/Heroku-like.
- data/dataset_estudiantes.csv: datos listos.

## Correr local
pip install -r requirements.txt
python dataset.py
python app.py  # http://127.0.0.1:8050

## Binder (guia)
1) Sube esta carpeta a un repo publico en GitHub.
2) Abre https://mybinder.org y pega tu repo.
3) Si el web app no abre por puerto, alternativa: crear un notebooks/dashboard.ipynb y usar Voila (agrega voila==0.5.6 a requirements).
