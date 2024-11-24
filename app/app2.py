from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Inicializar la aplicación Flask
app = Flask(__name__)

# Cargar el modelo previamente guardado
try:
    model = joblib.load('modelo_voting_classifier.pkl')  # Asegúrate de que el archivo esté en el mismo directorio
    print("Modelo cargado exitosamente.")
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    model = None

# Endpoint para predecir
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Modelo no cargado"}), 500

    try:
        # Obtener los datos de la solicitud en formato JSON
        data = request.get_json()

        # Asegurarse de que los datos contengan las claves necesarias
        required_keys = ["Gender", "Age", "Profession", "Academic Pressure", "Work Pressure", "Sleep Hours", "Suicidal Thoughts"]
        for key in required_keys:
            if key not in data:
                return jsonify({"error": f"Falta el campo {key}"}), 400

        # Convertir los datos a un DataFrame (esto puede depender de tu modelo y cómo lo entrenaste)
        input_data = pd.DataFrame([data])

        # Realizar la predicción
        prediction = model.predict(input_data)[0]  # Asegúrate de que tu modelo use esta forma de predicción

        # Devolver la predicción en formato JSON
        return jsonify({"prediction": prediction})

    except Exception as e:
        return jsonify({"error": f"Error al procesar la solicitud: {e}"}), 500

# Ruta principal para probar si la API está en funcionamiento
@app.route('/')
def home():
    return "La API está corriendo correctamente."

# Iniciar el servidor
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
