import numpy as np
import pandas as pd

def make_dataset(n=300, random_state=42):
    rng = np.random.default_rng(random_state)
    metodo = rng.choice(['A','B'], size=n)
    horas_estudio = np.clip(rng.normal(10, 3, n), 0, None)
    horas_suenio = np.clip(rng.normal(7, 1.2, n), 3, 12)
    indice_socioeco = np.clip(rng.normal(50, 10, n), 20, 90)
    gpa_previo = np.clip(rng.normal(3.2, 0.4, n), 0, 4.0)
    efecto_metodo = np.where(metodo=='A', 2.0, 0.0)
    ruido = rng.normal(0, 5, n)
    puntuacion = (4.0*horas_estudio + 1.5*horas_suenio + 0.3*indice_socioeco +
                  8.0*gpa_previo + efecto_metodo + ruido)
    puntuacion = np.clip(puntuacion, 0, 100)
    aprueba = (puntuacion >= 60).astype(int)
    return pd.DataFrame({
        'metodo': metodo,
        'horas_estudio': horas_estudio,
        'horas_suenio': horas_suenio,
        'indice_socioeco': indice_socioeco,
        'gpa_previo': gpa_previo,
        'puntuacion': puntuacion,
        'aprueba': aprueba
    })

if __name__ == "__main__":
    df = make_dataset()
    df.to_csv("data/dataset_estudiantes.csv", index=False)
    print("Dataset guardado en data/dataset_estudiantes.csv")
