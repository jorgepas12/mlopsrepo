import pytest
import joblib
import numpy as np

# Prueba para verificar la carga del modelo
def test_model_loading():
    model = joblib.load('model.pkl')
    assert model is not None, "El modelo no se cargó correctamente"

# Prueba para realizar una predicción de prueba
def test_model_prediction():
    model = joblib.load('model.pkl')
    
    # Datos de prueba simples
    test_data = np.array([8.3252, 41.0, 6.984127, 1.023809, 322.0, 2.555556, 37.88, -122.23]).reshape(1, -1)
    
    # Realizar predicción
    prediction = model.predict(test_data)
    
    # Validaciones básicas
    assert prediction is not None, "La predicción falló"
    assert prediction.shape == (1,), "La predicción debe ser un array unidimensional"
    assert isinstance(prediction[0], (int, float)), "La predicción debe ser un número"

# Prueba para verificar que el modelo es consistente (reproduce los mismos resultados)
def test_model_consistency():
    model = joblib.load('model.pkl')
    
    # Datos de prueba simples
    test_data = np.array([8.3252, 41.0, 6.984127, 1.023809, 322.0, 2.555556, 37.88, -122.23]).reshape(1, -1)

    # Realizar predicciones múltiples veces
    prediction_1 = model.predict(test_data)
    prediction_2 = model.predict(test_data)

    # Las predicciones deben ser idénticas
    assert np.array_equal(prediction_1, prediction_2), "El modelo debe producir predicciones consistentes"

# Prueba básica para verificar que el modelo no produce valores extremadamente altos o bajos
def test_model_output_range():
    model = joblib.load('model.pkl')
    
    # Datos de prueba simples
    test_data = np.array([8.3252, 41.0, 6.984127, 1.023809, 322.0, 2.555556, 37.88, -122.23]).reshape(1, -1)

    # Realizar predicción
    prediction = model.predict(test_data)

    # Verificar que la predicción está en un rango esperado (ajusta este rango según tu caso)
    assert 0.0 <= prediction[0] <= 10.0, f"La predicción está fuera del rango esperado: {prediction[0]}"
