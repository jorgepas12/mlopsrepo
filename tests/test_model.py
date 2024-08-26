import pytest
import joblib
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.dummy import DummyRegressor

# Prueba para verificar la carga del modelo
def test_model_loading():
    # Cargar el modelo entrenado
    model = joblib.load('model.pkl')
    assert model is not None, "El modelo no se cargó correctamente"

# Prueba para realizar una predicción de prueba
def test_model_prediction():
    # Cargar el modelo y realizar una predicción de prueba
    model = joblib.load('model.pkl')
    test_data = np.array([8.3252, 41.0, 6.984127, 1.023809, 322.0, 2.555556, 37.88, -122.23]).reshape(1, -1)
    prediction = model.predict(test_data)
    assert prediction is not None, "La predicción falló"
    assert prediction.shape == (1,), "La predicción debe ser un array unidimensional"
    assert isinstance(prediction[0], (int, float)), "La predicción debe ser un número"

# Prueba para verificar que el MSE del modelo es razonable
def test_model_performance():
    # Cargar el modelo entrenado
    model = joblib.load('model.pkl')
    
    # Datos de prueba simples
    X_test = np.array([[8.3252, 41.0, 6.984127, 1.023809, 322.0, 2.555556, 37.88, -122.23],
                       [6.4214, 21.0, 7.424127, 1.432809, 340.0, 3.556556, 35.45, -121.44']])
    y_test = np.array([2.5, 1.8])

    # Predicciones del modelo
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    # Ajusta el umbral de MSE según sea necesario
    assert mse < 5.0, f"El MSE es demasiado alto: {mse}"

# Prueba para comparar el modelo con un modelo base (DummyRegressor)
def test_model_vs_baseline():
    # Cargar el modelo entrenado
    model = joblib.load('model.pkl')
    
    # Datos de prueba simples
    X_test = np.array([[8.3252, 41.0, 6.984127, 1.023809, 322.0, 2.555556, 37.88, -122.23],
                       [6.4214, 21.0, 7.424127, 1.432809, 340.0, 3.556556, 35.45, -121.44']])
    y_test = np.array([2.5, 1.8])

    # Predicciones del modelo
    y_pred = model.predict(X_test)

    # Modelo base simple (predice la media)
    baseline_model = DummyRegressor(strategy="mean")
    baseline_model.fit(X_test, y_test)
    y_baseline_pred = baseline_model.predict(X_test)

    # Comparar MSE del modelo entrenado vs modelo base
    mse_model = mean_squared_error(y_test, y_pred)
    mse_baseline = mean_squared_error(y_test, y_baseline_pred)

    assert mse_model < mse_baseline, "El modelo entrenado debe superar al modelo base"

# Prueba para verificar que el modelo es consistente (reproduce los mismos resultados)
def test_model_consistency():
    # Cargar el modelo entrenado
    model = joblib.load('model.pkl')
    
    # Datos de prueba simples
    test_data = np.array([8.3252, 41.0, 6.984127, 1.023809, 322.0, 2.555556, 37.88, -122.23]).reshape(1, -1)

    # Realizar predicciones múltiples veces
    prediction_1 = model.predict(test_data)
    prediction_2 = model.predict(test_data)

    # Las predicciones deben ser idénticas
    assert np.array_equal(prediction_1, prediction_2), "El modelo debe producir predicciones consistentes"
