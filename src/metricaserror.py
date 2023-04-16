import numpy as np

import sklearn.metrics as skm

def R2(target, prediccion):
    return skm.r2_score(target, prediccion)


def MSE(target, prediccion):
    return skm.mean_squared_error(target, prediccion)


def MeanAE(target, prediccion):
    return skm.mean_absolute_error(target, prediccion)


def MedianAE(target, prediccion):
    return skm.median_absolute_error(target, prediccion)


def RMSE(target, prediccion):
    RMSE = np.power(target - prediccion, 2)
    RMSE = np.sqrt(RMSE)
    return np.sum(RMSE)


def MaxError(target, prediccion):
    return skm.max_error(target, prediccion)


def varianzaExplicada(target, prediccion):
    return skm.explained_variance_score(target, prediccion)


def MSLE(target, prediccion):
    return skm.mean_squared_log_error(target, prediccion)


def MAPE(target, prediccion):
    return skm.mean_absolute_percentage_error(target, prediccion)


def SMAPE(target, prediccion):
    SMAPE = (target - prediccion) / (np.abs(target) + np.abs(prediccion))
    SMAPE = np.sum(SMAPE)
    return np.mean(SMAPE) * 100


def calculo(metricas, target, prediccion):
    valor= []
    gbl= globals().items()
    funciones= {nombre: funcion for nombre, funcion in gbl if callable(funcion)}

    for metrica in metricas:
        valor.append(funciones[metrica](target, prediccion))

    return valor

if __name__ == '__main__':
    print(calculo(["SMAPE", "MeanAE", "RMSE"], np.array([5]), np.array([2])))