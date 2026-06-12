import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib

datos = pd.read_csv(
    "datos/estudiantes.csv"
)

X = datos[
    [
        "horas_estudio",
        "faltas",
        "trabajos_no_entregados"
    ]
]

y = datos["resultado"]

modelo = LogisticRegression()

modelo.fit(X, y)

joblib.dump(
    modelo,
    "modelos/modelo_aprobacion.pkl"
)

print("Modelo entrenado correctamente")