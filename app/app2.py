from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Cargar el modelo entrenado
modelo = joblib.load('modelo_entrenado.pkl')  # Ruta del modelo preentrenado

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Recibir los datos en formato JSON
        data = request.get_json()

        # Preprocesamiento: Convertir los datos a un formato adecuado para el modelo
        # Asegúrate de que los valores de las características están en el formato que el modelo espera
        features = [
            data['Gender'], 
            data['Age'], 
            data['Profession'], 
            data['Academic Pressure'],
            data['Work Pressure'], 
            data['Sleep Hours'], 
            data['Suicidal Thoughts']
        ]
        
        # Convertir características en el formato correcto para la predicción
        # Dependiendo del modelo, esto podría incluir transformaciones adicionales
        # Si el modelo espera que las variables categóricas estén codificadas, por ejemplo, con One-Hot Encoding,
        # deberías realizar esos cambios antes de hacer la predicción.

        # Realizar la predicción
        prediction_prob = modelo.predict_proba([features])[0][1]  # Usar probabilidad de la clase positiva

        # Retornar la probabilidad como respuesta
        return jsonify({"prediction": prediction_prob})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

