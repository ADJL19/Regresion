import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

import tensorflow as tf
from keras import layers

from modelosregresion import modelo
import funciones

path = "./data/output.csv"
[etiquetas, predictores] = funciones.importacionDatos(path)

# scoring = ['max_error', 'neg_mean_absolute_error', 'neg_root_mean_squared_error', 'neg_median_absolute_error', 'r2']
# errores = ['Error máximo', 'MeanAE', 'MSE', 'MedianAE', 'Coeficiente de correlación']
metricas = dict(Maxerror="max_error", MeanAE="neg_mean_absolute_error", MSE="neg_root_mean_squared_error", MedianAE="neg_median_absolute_error", coeficienteCorrelación="r2")

# modelos = {}
# modelos['RL'] = modelo(LinearRegression())
# modelos['KNN'] = modelo(KNeighborsRegressor())
# modelos['SVM'] = modelo(SVR())
# modelos['DT'] = modelo(DecisionTreeRegressor())

# funciones.validacionCruzada(modelos, predictores, etiquetas, metricas)
# df1 = funciones.crearDF(modelos, metricas)
# funciones.boxplot(data= df1, metrica= "RMSE")

modelos = {}
modelos['RL1'] = modelo(LinearRegression(fit_intercept= False))
modelos['RL2'] = modelo(LinearRegression(fit_intercept= True))
funciones.validacionCruzada(modelos, predictores, etiquetas, metricas)
df1 = funciones.crearDF(modelos, metricas)
print(df1)
funciones.scatter(data= df1, metrica= "MeanAE")
funciones.boxplot(data= df1, metrica= "MeanAE")