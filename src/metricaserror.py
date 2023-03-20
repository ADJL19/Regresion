import numpy as np

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def R2(etiquetas, prediccion):
    return r2_score(etiquetas, prediccion)


def MSE(etiquetas, prediccion):
    # MSE = np.power(etiquetas - prediccion, 2)
    # return np.mean(MSE)

    return mean_squared_error(etiquetas, prediccion)


def MAE(etiquetas, prediccion):
    # MAE = etiquetas - prediccion
    # MAE = np.abs(MAE))
    # return np.mean(MAE)

    return mean_absolute_error(etiquetas, prediccion)


def RMSE(etiquetas, prediccion):
    RMSE = np.power(etiquetas - prediccion, 2)
    RMSE = np.sqrt(RMSE)
    return np.sum(RMSE)


def MAPE(etiquetas, prediccion):
    MAPE = (etiquetas - prediccion) / etiquetas
    MAPE = np.sum(MAPE)
    return np.mean(MAPE) * 100


def SMAPE(etiquetas, prediccion):
    SMAPE = (etiquetas - prediccion) / (np.abs(etiquetas) + np.abs(prediccion))
    SMAPE = np.sum(SMAPE)
    return np.mean(SMAPE) * 100
