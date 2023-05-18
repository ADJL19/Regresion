"""Script para el cálculo de las métricas de error."""

#Importación de las librerías usadas.
import numpy as np
import sklearn.metrics as skm

def R2(target, prediccion):
    """
    Calcula el coeficiente de determinación.

    Parámetros:
    ----------
    target: array-like de los valores de los valores reales.
    prediccion: array-like de los valores predichos.

    Devuelve:
    ----------
    r: número flotante con el coeficiente de determinación.
    """
    return skm.r2_score(target, prediccion)


def MSE(target, prediccion):
    """
    Calcula el error cuadrático medio.

    Parámetros:
    ----------
    target: array-like de los valores de los valores reales.
    prediccion: array-like de los valores predichos.

    Devuelve:
    ----------
    r: número flotante con el error cuadrático medio.
    """
    return skm.mean_squared_error(target, prediccion)


def MeanAE(target, prediccion):
    """
    Calcula el error absoluto medio.

    Parámetros:
    ----------
    target: array-like de los valores de los valores reales.
    prediccion: array-like de los valores predichos.

    Devuelve:
    ----------
    r: número flotante con el error absoluto medio.
    """
    return skm.mean_absolute_error(target, prediccion)


def MedianAE(target, prediccion):
    return skm.median_absolute_error(target, prediccion)


def RMSE(target, prediccion):
    return skm.mean_squared_error(target, prediccion, squared= False)


def MaxError(target, prediccion):
    return skm.max_error(target, prediccion)


def varianzaExplicada(target, prediccion):
    return skm.explained_variance_score(target, prediccion)


def MSLE(target, prediccion):
    return skm.mean_squared_log_error(target, prediccion)


def MAPE(target, prediccion):
    return skm.mean_absolute_percentage_error(target, prediccion)


def SMAPE(target, prediccion):
    prediccion = np.reshape(prediccion, -1)
    SMAPE = (target - prediccion) / (np.abs(target) + np.abs(prediccion))
    SMAPE = np.sum(SMAPE)
    return np.mean(SMAPE) * 100


def calculo(metricas, target, prediccion):
    """
    Realiza los cálculos de las métricas indicadas.
    
    Parámetros:
    ----------
    metricas: Array-like de las métricas a calcular
    target: Array-like de los valores de los valores reales.
    prediccion: Array-like de los valores predichos.

    Devuelve:
    ----------
    n: Array-like con el valor de las métricas.
    """
    valor= []
    gbl= globals().items()
    funciones= {nombre: funcion for nombre, funcion in gbl if callable(funcion)}

    for metrica in metricas:
        valor.append(funciones[metrica](target, prediccion))

    return np.array(valor)