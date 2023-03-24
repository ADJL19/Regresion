#Importación de las librerías empleadas.
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from modelosregresion import modelo
import funciones

#Establecemos la ruta donde se encuentran los datos y los importamos.
path = "./data/2017-18_meteoYeolica.csv"
data = pd.read_csv(path)

[target, predictores] = funciones.importacionDatos(path)

#Generamos un diccionario donde se introducen las métricas que se desean evaluar.
metricas = dict(Maxerror="max_error", MeanAE="neg_mean_absolute_error", RMSE="neg_root_mean_squared_error", MedianAE="neg_median_absolute_error", coeficienteCorrelación="r2")

#Generamos un diccionario donde se almacenan los distintos modelos que se van a emplear.
modelos = {}
modelos['RL'] = modelo(LinearRegression())
modelos['KNN'] = modelo(KNeighborsRegressor())
# modelos['SVM'] = modelo(SVR())
modelos['DT'] = modelo(DecisionTreeRegressor())

#Realizamos la validación cruzada de los modelos.
funciones.validacionCruzada(modelos, predictores, target, metricas)
#Creamos un DataFrame con todas las métricas. Este DF se organiza con las distintas métricas en columnas, junto a un indicador del modelo
#y de la iteración
df1 = funciones.crearDF(modelos, metricas)

#Graficamos boxplot de distintas métricas
funciones.variasBoxplot(modelos, df1, "MeanAE", "coeficienteCorrelación", "MedianAE")

#Graficamos una figura de dispersión para ver la variaión de ciertas métricas en las iteraciones
funciones.variasScatter(modelos, df1, "MeanAE", "RMSE")

#Escribimos el valor de las métricas en el excel, en la hoja 'validacionCruzada'
df1.to_excel("./Regresion.xlsx", sheet_name= "Validación cruzada")

#Para estudiar el impacto de los hiperparámetros, se crean modelos con distintas inicializaciones:
modelosRL = {}
modelos['RL0'] = modelo(LinearRegression())
modelos['RL1'] = modelo(LinearRegression(fit_intercept= True))
funciones.validacionCruzada(modelosRL, predictores, target, metricas)
df1 = funciones.crearDF(modelosRL, metricas)
df1.to_excel("./Regresion.xlsx", sheet_name= "Regresion Lineal")

modelosKNN = {}
modelos['KN0'] = modelo(KNeighborsRegressor())
modelos['KN1'] = modelo(KNeighborsRegressor(n_neighbors= 2))
modelos['KN2'] = modelo(KNeighborsRegressor(n_neighbors= 10))
modelos['KN3'] = modelo(KNeighborsRegressor(weights= 'distance'))
modelos['KN4'] = modelo(KNeighborsRegressor(n_neighbors= 2, weights= 'distance'))
modelos['KN5'] = modelo(KNeighborsRegressor(n_neighbors= 10, weights= 'distance'))
funciones.validacionCruzada(modelosKNN, predictores, target, metricas)
df1 = funciones.crearDF(modelosKNN, metricas)
df1.to_excel("./Regresion.xlsx", sheet_name= "K vecinos más cercanos")

modelosSVM = {}
modelosSVM['SVM0'] = modelo(SVR())
modelosSVM['SVM1'] = modelo(SVR(kernel= 'poly', kerneldegree= 5))
modelosSVM['SVM2'] = modelo(SVR(kernel= 'poly', kerneldegree= 7))
funciones.validacionCruzada(modelosSVM, predictores, target, metricas)
df1 = funciones.crearDF(modelosSVM, metricas)
df1.to_excel("./Regresion.xlsx", sheet_name= "Máquina de vectores soporte")

modelosDT = {}
modelosDT['DT0'] = modelos(DecisionTreeRegressor())
modelosDT['DT1'] = modelos(DecisionTreeRegressor(criterion= 'poison'))
modelosDT['DT2'] = modelos(DecisionTreeRegressor(criterion= 'absolute_error'))
modelosDT['DT3'] = modelos(DecisionTreeRegressor(splitter= 'random'))
funciones.validacionCruzada(modelosDT, predictores, target, metricas)
df1 = funciones.crearDF(modelosDT, metricas)
df1.to_excel("./Regresion.xlsx", sheet_name= "Árbol de decisión")



# hola0 = df1.MeanAE[df1.Tecnica == 0].to_numpy()
# hola1 = df1.MeanAE[df1.Tecnica == 1].to_numpy()
# hola2 = df1.MeanAE[df1.Tecnica == 2].to_numpy()
# hola3 = df1.MeanAE[df1.Tecnica == 3].to_numpy()
# funciones.contrasteHipotesis(hola0, hola1, hola2, hola3, alpha= 0.05)