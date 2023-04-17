#Importación de las librerías empleadas.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
import sklearn.metrics as skm
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA, FastICA
import scipy.stats as stats
from statsmodels.stats.multicomp import MultiComparison

from modelosregresion import modelo
import funciones
import redNeuronal

from timeit import timeit



pca= PCA(n_components= 4)
ica= FastICA(n_components= 4)

#Establecemos la ruta donde se encuentran los datos y los importamos.
path = "C:/Users/adzl/Desktop/BI/Enerxia/MiniEolica/Datos/2017-18_meteoYeolica.csv"

#Importamos los valores de predicores y target, indicando si queremos normalizar los datos o aplicar reducción de dimensionalidad
[target, predictores] = funciones.importacionDatos(path, normalizar= True, reduccion= ica)
# funciones.representarDatos(pd.concat([predictores, target], axis=1, join="inner"))

p_train, p_test, t_train, t_test= train_test_split(predictores, target, test_size= 0.1, shuffle= False)

#Generamos un diccionario donde se introducen las métricas que se desean evaluar.
metricas = dict(MaxError= "max_error", MSE= "neg_mean_squared_error", MeanAE= "neg_mean_absolute_error", RMSE= "neg_root_mean_squared_error", MedianAE= "neg_median_absolute_error", R2= "r2", varianzaExplicada= "explained_variance")
test = list(metricas.keys())

# #Generamos un diccionario donde se almacenan los distintos modelos que se van a emplear.
# modelos = {}
# modelos['RL'] = modelo(LinearRegression())
# modelos['KNN'] = modelo(KNeighborsRegressor())
# modelos['SVM'] = modelo(SVR())
# modelos['DT'] = modelo(DecisionTreeRegressor())

# #Realizamos la validación cruzada de los modelos.
# funciones.validacionCruzada(modelos, predictores, target, metricas)
# #Creamos un DataFrame con todas las métricas. Este DF se organiza con las distintas métricas en columnas, junto a un indicador del modelo
# #y de la iteración
# dfM = funciones.crearDF(modelos, metricas)
# funciones.boxplot(dfM, "RMSE")

# #Realizamos la validación cruzada de la red neuronal.
# dfR = redNeuronal.validacionCruzada(predictores, target)
# #Concatenamos los dos DF obtenidos en un único DF.
# dfM = pd.concat([dfM, dfR])

# #Graficamos boxplot de distintas métricas
# funciones.variasBoxplot(modelos, dfM, "MeanAE", "RMSE")
# #Graficamos una figura de dispersión para ver la variación de ciertas métricas en las iteraciones
# funciones.variasScatter(modelos, dfM, "MeanAE", "RMSE")
# #Escribimos el valor de las métricas en el excel, en la hoja 'Validación cruzada'
# dfM.to_excel("./Regresion.xlsx", sheet_name= "Validación cruzada")



# modelosRL = {}
# modelosRL['RL00'] = modelo(LinearRegression())
# modelosRL['RL01'] = modelo(LinearRegression(fit_intercept= False))
# modelosRL['RL02'] = modelo(LinearRegression(positive= True))
# modelosRL['RL03'] = modelo(LinearRegression(fit_intercept= False, positive= True))

# dfM = funciones.validacionCruzada(modelosRL, predictores, target, metricas)
# funciones.variasBoxplot(dfM, "MSE", "RMSE", "MeanAE", "MedianAE", "R2")
# dfM.to_excel("./info/metricas/RLin.xlsx", sheet_name= "Regresion Lineal")

# modelosRL['RL00'].entrenarModelo(p_train, t_train)
# t_pred = modelosRL['RL00'].predecir(p_test)
# display= skm.PredictionErrorDisplay.from_predictions(y_true= t_test, y_pred= t_pred, kind= 'actual_vs_predicted')
# plt.show()



# modelosKNN = {}
# modelosKNN['KNN00'] = modelo(KNeighborsRegressor())
# modelosKNN['KNN01'] = modelo(KNeighborsRegressor(n_neighbors= 2))
# modelosKNN['KNN04'] = modelo(KNeighborsRegressor(n_neighbors= 30))
# modelosKNN['KNN05'] = modelo(KNeighborsRegressor(n_neighbors= 50))
# modelosKNN['KNN10'] = modelo(KNeighborsRegressor(weights= 'distance'))
# modelosKNN['KNN11'] = modelo(KNeighborsRegressor(n_neighbors= 2, weights= 'distance'))
# modelosKNN['KNN14'] = modelo(KNeighborsRegressor(n_neighbors= 30, weights= 'distance'))
# modelosKNN['KNN15'] = modelo(KNeighborsRegressor(n_neighbors= 50, weights= 'distance'))
# dfM = funciones.validacionCruzada(modelosKNN, predictores, target, metricas)
# funciones.variasBoxplot(dfM, "MSE", "RMSE", "MeanAE", "MedianAE", "R2")
#dfM.to_excel("./info/metricas/KNN.xlsx", sheet_name= "K vecinos más cercanos")

# modelosKNN['KNN04'].entrenarModelo(p_train, t_train)
# t_pred = modelosKNN['KNN04'].predecir(p_test)
# display= skm.PredictionErrorDisplay.from_predictions(y_true= t_test, y_pred= t_pred, kind= 'actual_vs_predicted')
# plt.show()



# modelosSVM = {}
# modelosSVM['SVM00'] = modelo(SVR(C= 0.8, tol= 0.01, cache_size= 2000), CV= 20)
# modelosSVM['SVM01'] = modelo(SVR(C= 0.8, tol= 0.01, kernel= 'linear', cache_size= 2000), CV= 20)
# modelosSVM['SVM02'] = modelo(SVR(C= 0.8, tol= 0.01, kernel= 'sigmoid', cache_size= 2000), CV= 20)
# modelosSVM['SVM10'] = modelo(SVR(C= 0.8, tol= 0.01, kernel= 'poly', cache_size= 2000), CV= 20)
# modelosSVM['SVM11'] = modelo(SVR(C= 0.8, tol= 0.01, kernel= 'poly', degree= 5, cache_size= 2000), CV= 20)
# modelosSVM['SVM12'] = modelo(SVR(C= 0.8, tol= 0.01, kernel= 'poly', degree= 7, cache_size= 2000), CV= 20)
# funciones.validacionCruzada(modelosSVM, predictores, target, metricas)
# df1 = funciones.crearDF(modelosSVM, metricas)
# dfM.to_excel("./info/metricas/SVM.xlsx", sheet_name= "Máquina de vectores soporte")



# modelosDT = {}
# modelosDT['DT00'] = modelo(DecisionTreeRegressor(criterion= 'squared_error'))
# modelosDT['DT01'] = modelo(DecisionTreeRegressor(criterion= 'squared_error', max_depth= 5))
# modelosDT['DT02'] = modelo(DecisionTreeRegressor(criterion= 'squared_error', max_depth= 10))
# modelosDT['DT03'] = modelo(DecisionTreeRegressor(criterion= 'squared_error', max_depth= 12))
# modelosDT['DT10'] = modelo(DecisionTreeRegressor(criterion= 'absolute_error'))
# modelosDT['DT11'] = modelo(DecisionTreeRegressor(criterion= 'absolute_error', max_depth= 5))
# modelosDT['DT12'] = modelo(DecisionTreeRegressor(criterion= 'absolute_error', max_depth= 10))
# modelosDT['DT13'] = modelo(DecisionTreeRegressor(criterion= 'absolute_error', max_depth= 12))
# dfM = funciones.validacionCruzada(modelosDT, predictores, target, metricas)
# funciones.variasBoxplot(dfM, "MSE", "RMSE", "MeanAE", "MedianAE", "R2")
# dfM.to_excel("./info/metricas/DT.xlsx", sheet_name= "Árbol de decisión")

# modelosDT['DT02'].entrenarModelo(p_train, t_train)
# t_pred = modelosDT['DT02'].predecir(p_test)
# display= skm.PredictionErrorDisplay.from_predictions(y_true= t_test, y_pred= t_pred, kind= 'actual_vs_predicted')
# plt.show()



# modelosRF = {}
# modelosRF['RF00'] = modelo(RandomForestRegressor(criterion= 'squared_error', n_estimators= 10, n_jobs= -1))
# modelosRF['RF01'] = modelo(RandomForestRegressor(criterion= 'squared_error', max_depth= 5, n_estimators= 10, n_jobs= -1))
# modelosRF['RF02'] = modelo(RandomForestRegressor(criterion= 'squared_error', max_depth= 10, n_estimators= 10, n_jobs= -1))
# modelosRF['RF202'] = modelo(RandomForestRegressor(criterion= 'squared_error', max_depth= 10, n_estimators= 20, n_jobs= -1))
# modelosRF['RF502'] = modelo(RandomForestRegressor(criterion= 'squared_error', max_depth= 10, n_estimators= 50, n_jobs= -1))
# modelosRF['RF03'] = modelo(RandomForestRegressor(criterion= 'squared_error', max_depth= 12, n_estimators= 10, n_jobs= -1))
# modelosRF['RF10'] = modelo(RandomForestRegressor(criterion= 'absolute_error', n_estimators= 10, n_jobs= -1))
# modelosRF['RF11'] = modelo(RandomForestRegressor(criterion= 'absolute_error', max_depth= 5, n_estimators= 10, n_jobs= -1))
# modelosRF['RF12'] = modelo(RandomForestRegressor(criterion= 'absolute_error', max_depth= 10, n_estimators= 10, n_jobs= -1))
# modelosRF['RF212'] = modelo(RandomForestRegressor(criterion= 'absolute_error', max_depth= 10, n_estimators= 20, n_jobs= -1))
# modelosRF['RF512'] = modelo(RandomForestRegressor(criterion= 'absolute_error', max_depth= 10, n_estimators= 50, n_jobs= -1))
# modelosRF['RF13'] = modelo(RandomForestRegressor(criterion= 'absolute_error', max_depth= 12, n_estimators= 10, n_jobs= -1))

# dfM = funciones.validacionCruzada(modelosRF, predictores, target, metricas)
# funciones.variasBoxplot(dfM, "MSE", "RMSE", "MeanAE", "MedianAE", "R2")
# dfM.to_excel("./info/metricas/RF.xlsx", sheet_name= "Bosque de decisión")

# modelosRF['RF502'].entrenarModelo(p_train, t_train)
# t_pred = modelosRF['RF502'].predecir(p_test)
# display= skm.PredictionErrorDisplay.from_predictions(y_true= t_test, y_pred= t_pred, kind= 'actual_vs_predicted')
# plt.show()



# dfR1 = redNeuronal.validacionCruzada(predictores, target, NO=[10, 10])
# dfR2 = redNeuronal.validacionCruzada(predictores, target, NO=[15, 15])
# dfR3 = redNeuronal.validacionCruzada(predictores, target, NO=[10, 10], FA='sigmoid')



# #Realizamos el contraste de hipótesis:
# print("")
# alpha = 0.05

# dRL = modelos['RL'].scores['test_'+ metricas['MeanAE']]
# dKNN = modelos['KNN'].scores['test_'+ metricas['MeanAE']]
# dDT = modelos['DT'].scores['test_'+ metricas['MeanAE']]

# F_statistic, pVal = stats.kruskal(dRL, dKNN, dDT)
# print ('p-valor KrusW:', pVal)
# if pVal <= alpha:
#     print('Rechazamos la hipótesis: los modelos son diferentes\n')
#     stacked_data = dfM.MeanAE
#     stacked_model = dfM.Tecnica
#     MultiComp = MultiComparison(stacked_data, stacked_model)  
#     print(MultiComp.tukeyhsd(alpha=0.05))
# else:
#     print('Aceptamos la hipótesis: los modelos son iguales')