import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

# Función para cargar los datos
def load_data():
    try:
        housing = fetch_california_housing()
        X = pd.DataFrame(housing.data, columns=housing.feature_names)
        y = pd.Series(housing.target, name='MedHouseVal')
        return X, y
    except Exception as e:
        print(f"Error al cargar los datos: {e}")
        raise

# Validar la integridad de los datos
def validate_data(X, y):
    assert X.isnull().sum().sum() == 0, "Los datos contienen valores nulos"
    assert len(X) == len(y), "El número de muestras y etiquetas no coincide"
    assert X.shape[1] == 8, f"Se esperaban 8 características, pero se encontraron {X.shape[1]}"

# Función para dividir los datos
def split_data(X, y):
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test
    except Exception as e:
        print(f"Error al dividir los datos: {e}")
        raise

# Función para entrenar el modelo
def train_model(X_train, y_train):
    try:
        model = LinearRegression()
        model.fit(X_train, y_train)
        return model
    except Exception as e:
        print(f"Error al entrenar el modelo: {e}")
        raise

# Función para evaluar el modelo
def evaluate_model(model, X_test, y_test):
    try:
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"Mean Squared Error: {mse}")
        print(f"R^2 Score: {r2}")
        assert mse < 1.0, "El MSE es demasiado alto, considera mejorar el modelo"
        assert r2 > 0.5, "El R^2 Score es bajo, considera mejorar el modelo"
    except Exception as e:
        print(f"Error al evaluar el modelo: {e}")
        raise

# Función para guardar el modelo
def save_model(model, filepath='model.pkl'):
    try:
        joblib.dump(model, filepath)
        assert os.path.exists(filepath), f"No se pudo guardar el modelo en {filepath}"
        print(f"Modelo guardado exitosamente en {filepath}")
    except Exception as e:
        print(f"Error al guardar el modelo: {e}")
        raise

# Flujo principal
if __name__ == "__main__":
    try:
        # Cargar y validar los datos
        X, y = load_data()
        validate_data(X, y)

        # Dividir los datos
        X_train, X_test, y_train, y_test = split_data(X, y)

        # Entrenar el modelo
        model = train_model(X_train, y_train)

        # Evaluar el modelo
        evaluate_model(model, X_test, y_test)

        # Guardar el modelo
        save_model(model)

    except Exception as e:
        print(f"Fallo en el proceso de entrenamiento: {e}")

