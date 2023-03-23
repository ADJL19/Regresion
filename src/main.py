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

path = "./data/output.csv"
[etiquetas, predictores] = datos.importacionDatos(path)

scoring = ['max_error', 'neg_mean_absolute_error', 'neg_root_mean_squared_error', 'neg_median_absolute_error', 'r2']
errores = ['Error máximo', 'MAE', 'MSE', 'MAE', 'Coeficiente de correlación']

RL = modelo(LinearRegression())
KNN = modelo(KNeighborsRegressor())
SVM = modelo(SVR())
DT = modelo(DecisionTreeRegressor())

modelos = []
modelos.append(RL)
# modelos.append(KNN)
# modelos.append(SVM)
# modelos.append(DT)

for nombre, tecnica in zip(["Regresión Lineal", "K vecinos más cercanos", "SVM", "Árbol de decisión"], modelos):
    print(nombre)
    scores = tecnica.validacionCruzada(predictores, etiquetas)
    for test, valor in zip(errores, scoring):
        print(f"El {test} vale {np.mean(scores['test_' + valor])}")
    print("")

MAE = (RL.scores)
nombre = np.ones((10, 1))
nModelo = np.arange(1, 11)

datos = np.column_stack((MAE, nombre, nModelo))

df = pd.DataFrame(datos, columns= ['MAE', 'Metodo', 'Iteracion'])
graficas.boxplot(data= df, x= 'Iteracion', y= 'MAE')