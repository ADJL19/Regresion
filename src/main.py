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
modelos['SVM'] = modelo(SVR())
modelos['DT'] = modelo(DecisionTreeRegressor())

#Realizamos la validación cruzada de los modelos.
funciones.validacionCruzada(modelos, predictores, target, metricas)
#Creamos un DataFrame con todas las métricas. Este DF se organiza con las métricas en columnas
df1 = funciones.crearDF(modelos, metricas)
df1.to_excel("./Regresion.xlsx", sheet_name= "validacionCruzada")
funciones.boxplot(data= df1, metrica= "MeanAE")
funciones.boxplot(data= df1, metrica= "coeficienteCorrelación")
funciones.boxplot(data= df1, metrica= "MedianAE")

funciones.scatter(data= df1, metrica= "MeanAE")
funciones.scatter(data= df1, metrica= "RMSE")



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