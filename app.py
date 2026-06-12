from flask import Flask
from flask import render_template
from flask import request
import joblib

app = Flask(__name__)

modelo = joblib.load(
    "modelos/modelo_aprobacion.pkl"
)

@app.route("/")
def inicio():
    return render_template(
        "index.html"
    )

@app.route(
    "/predecir",
    methods=["POST"]
)
def predecir():

    horas = float(
        request.form["horas"]
    )

    faltas = float(
        request.form["faltas"]
    )

    trabajos = float(
        request.form["trabajos"]
    )

    estudiante = [
        [
            horas,
            faltas,
            trabajos
        ]
    ]

    resultado = modelo.predict(
        estudiante
    )

    probabilidad = modelo.predict_proba(
        estudiante
    )

    if resultado[0] == 1:
        mensaje = "APROBARÁ"
    else:
        mensaje = "REPROBARÁ"

    return render_template(
        "index.html",
        mensaje=mensaje,
        probabilidad=round(
            probabilidad[0][1] * 100,
            2
        )
    )

if __name__ == "__main__":
    app.run(
        debug=True
    )