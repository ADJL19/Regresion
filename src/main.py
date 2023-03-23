import numpy as np
import pandas as pd

import datos
import graficas

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

import tensorflow as tf
from keras import layers

from modelosregresion import modelo
from funciones import crearDF

path = "./data/output.csv"
[etiquetas, predictores] = datos.importacionDatos(path)

scoring = ['max_error', 'neg_mean_absolute_error', 'neg_root_mean_squared_error', 'neg_median_absolute_error', 'r2']
errores = ['Error máximo', 'MAE', 'MSE', 'MAE', 'Coeficiente de correlación']

RL = modelo(LinearRegression())
KNN = modelo(KNeighborsRegressor())
SVM = modelo(SVR())
DT = modelo(DecisionTreeRegressor())

modelos = {}
modelos['RL'] = RL
modelos['KNN'] = KNN
modelos['SVM'] = SVM
modelos['DT'] = DT

for nombre, tecnica in modelos.items():
    print(nombre)
    scores = tecnica.validacionCruzada(predictores, etiquetas)
    for test, valor in zip(errores, scoring):
        print(f"El {test} vale {np.mean(scores['test_' + valor])}")
    print("")

df = crearDF(modelos, scoring)
print(df)