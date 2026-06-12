import joblib

modelo = joblib.load(
    "modelos/modelo_aprobacion.pkl"
)

estudiante = [
    [10, 2, 0]
]

resultado = modelo.predict(
    estudiante
)

probabilidad = modelo.predict_proba(
    estudiante
)

print("Resultado:", resultado[0])
print(
    "Probabilidad:",
    round(probabilidad[0][1] * 100, 2),
    "%"
)