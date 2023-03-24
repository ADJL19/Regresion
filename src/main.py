import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from modelosregresion import modelo
import funciones

path = "./data/output.csv"
[etiquetas, predictores] = funciones.importacionDatos(path)

metricas = dict(Maxerror="max_error", MeanAE="neg_mean_absolute_error", MSE="neg_root_mean_squared_error", MedianAE="neg_median_absolute_error", coeficienteCorrelación="r2")

modelos = {}
modelos['RL'] = modelo(LinearRegression())
modelos['KNN'] = modelo(KNeighborsRegressor())
modelos['SVM'] = modelo(SVR())
modelos['DT'] = modelo(DecisionTreeRegressor())

funciones.validacionCruzada(modelos, predictores, etiquetas, metricas)
df1 = funciones.crearDF(modelos, metricas)
df1.to_excel("./Regresion.xlsx", sheet_name= "validacionCruzada")
# funciones.boxplot(data= df1, metrica= "MeanAE")

# modelos = {}
# modelos['RL1'] = modelo(LinearRegression(fit_intercept= False))
# modelos['RL2'] = modelo(LinearRegression(fit_intercept= True))
# funciones.validacionCruzada(modelos, predictores, etiquetas, metricas)
# df1 = funciones.crearDF(modelos, metricas)

# funciones.scatter(data= df1, metrica= "MeanAE")
# funciones.boxplot(data= df1, metrica= "MeanAE")

# hola0 = df1.MeanAE[df1.Tecnica == 0].to_numpy()
# hola1 = df1.MeanAE[df1.Tecnica == 1].to_numpy()
# hola2 = df1.MeanAE[df1.Tecnica == 2].to_numpy()
# hola3 = df1.MeanAE[df1.Tecnica == 3].to_numpy()

# funciones.contrasteHipotesis(hola0, hola1, hola2, hola3, alpha= 0.05)