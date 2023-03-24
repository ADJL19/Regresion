#Importación de las librerías empleadas.
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from modelosregresion import modelo
import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd, MultiComparison
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

# #Para estudiar el impacto de los hiperparámetros, se crean modelos con distintas inicializaciones:
modelosRL = {}
modelosRL['RL0'] = modelo(LinearRegression())
modelosRL['RL1'] = modelo(LinearRegression(fit_intercept= True))
funciones.validacionCruzada(modelosRL, predictores, target, metricas)
df1 = funciones.crearDF(modelosRL, metricas)
df1.to_excel("./Regresion.xlsx", sheet_name= "Regresion Lineal")

modelosKNN = {}
modelosKNN['KN0'] = modelo(KNeighborsRegressor())
modelosKNN['KN1'] = modelo(KNeighborsRegressor(n_neighbors= 2))
modelosKNN['KN2'] = modelo(KNeighborsRegressor(n_neighbors= 10))
modelosKNN['KN3'] = modelo(KNeighborsRegressor(weights= 'distance'))
modelosKNN['KN4'] = modelo(KNeighborsRegressor(n_neighbors= 2, weights= 'distance'))
modelosKNN['KN5'] = modelo(KNeighborsRegressor(n_neighbors= 10, weights= 'distance'))
funciones.validacionCruzada(modelosKNN, predictores, target, metricas)
df1 = funciones.crearDF(modelosKNN, metricas)
df1.to_excel("./Regresion.xlsx", sheet_name= "K vecinos más cercanos")

modelosSVM = {}
modelosSVM['SVM0'] = modelo(SVR())
modelosSVM['SVM1'] = modelo(SVR(kernel= 'poly'))
modelosSVM['SVM2'] = modelo(SVR(kernel= 'poly', degree= 7))
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




#Realizamos la contraste de hipótesis:
print("")
alpha = 0.05

dRL = modelos['RL'].scores['test_'+ metricas['MeanAE']]
dKNN = modelos['KNN'].scores['test_'+ metricas['MeanAE']]
dDT = modelos['DT'].scores['test_'+ metricas['MeanAE']]

F_statistic, pVal = stats.kruskal(dRL, dKNN, dDT)
print ('p-valor KrusW:', pVal)
if pVal <= alpha:
    print('Rechazamos la hipótesis: los modelos son diferentes\n')
    stacked_data = df1.MeanAE
    stacked_model = df1.Tecnica
    MultiComp = MultiComparison(stacked_data, stacked_model)  
    print(MultiComp.tukeyhsd(alpha=0.05))
else:
    print('Aceptamos la hipótesis: los modelos son iguales')