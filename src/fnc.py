#Importación de las librerías empleadas
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd

from claseModelos import model

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

import gph
import metricaserror as error

def importacionDatos(config, reduccion= None):
    """
    Importa los datos de archivos Excel o CSV.

    Parámetros:
    ----------
    config: Archivo JSON donde se establece la ruta del archivo, su extensión y el separador en caso de ser archivos CSV. Además, en este archivo, se establece si se realizará la normalización de los datos o si se aplicará algún método de reducción de dimensionalidad.

    Devuelve:
    ----------
    target: Dataframe con las variables a predecir y su valor.
    predictores: Dataframe con las variables explicativas y su valor.
    """

    archivo= {'.csv': pd.read_csv, '.xlsx': pd.read_excel}
    path= config['fichero_datos']['fichero'] + config['fichero_datos']['extension']

    #Leemos el archivo donde se almacenan los datos
    data = archivo[config['fichero_datos']['extension']](path, sep= config['fichero_datos']['separador'])

    # #Eliminamos los datos espurios
    # data = data[data['Enerxia'] > 0]
    # data = data[data['Enerxia'] < 1500]
    # data = data[data['Velocidade'] < 14]
    # data = data[data['I'] > 0]
    # data = data[data['Velocidade'] > 3.5]

    #Realizamos el reseteo del DF para obtener sus índices ordenados desde 0 hasta N,
    #en lugar de desde 0 hasta el máximo original, pero con tan solo N índices.
    data = pd.concat([data], ignore_index= True)

    #Se aleatorizan los datos.
    if config["fichero_datos"]["aleatorizar"]: data = data.sample(n = len(data))

    #Se dividen los datos en target y predictores
    target = data.Energy

    #Se eliminan del DF las variables no útiles.
    data = data.drop(columns= ['Time', 'Energy', 'V', 'I', 'W', 'VAr', 'Wh_e'])
    
    # gph.matrizCorrelacion(pd.concat([data, target], axis=1, join="inner"))

    #Se normalizan los datos si así de indica.
    if config["fichero_datos"]["normalizar"]: data= pd.DataFrame(normalizacion(data), columns= data.columns)

    #Se realiza la reducción introducida:
    if reduccion!= None:
        predictores= reduccion.fit_transform(data)

        variablesUsadas= vReduccion(reduccion, data)
        predictores = pd.DataFrame(predictores, columns= variablesUsadas)
    else:
        predictores = data.loc[:, ['Wind_V', 'Wind_V_10m', 'SD_Wind_V_10m', 'SD_Wind_D_10m']]
        # predictores = data.loc[:, ['Velocidade', 'VelocidadeDoVentoA10m', 'DesviacionTi_picaDaVelocidadeDoVentoA10m', 'DesviacionTipicaDaDireccionDoVentoA10m']]

    return target, predictores

def vReduccion(model, data):
    nComp= len(model.components_)
    mejorExplicacion = [np.abs(model.components_[comp]).argmax() for comp in range(nComp)]
    variables = data.columns
    return [variables[mejorExplicacion[comp]] for comp in range(nComp)]

def normalizacion(data):
    scaler = StandardScaler()
    scaler.fit(data)
    return scaler.transform(data)

#Función que realiza la validación cruzada de los distintos modelos.
def validacionCruzada(modelos, predictores, target, metricas):
    #Se recorre el diccionario donde se encuentran los modelos.
    for nombre, tecnica in modelos.items():
        validacionCruzadaSimple(nombre, tecnica, predictores, target, metricas)

    return crearDF(modelos, metricas)

def validacionCruzadaSimple(nombre, tecnica, predictores, target, metricas):
    print(f"Para el modelo {nombre} con parámetros {tecnica.parametros}:")
    tecnica.validacionCruzada(predictores, target, list(metricas.values()))
    for test, valor in metricas.items():
        print(f"El {test} vale {np.mean(tecnica.scores['test_' + valor])}")
    print("")

def predValidacionCruzada(modelos, predictores, target, metricas):
    n = 0
    col = metricas.copy()
    col.append('Iteracion')
    df1 = pd.DataFrame(columns= col)
    df2 = pd.DataFrame(columns= ['Tecnica'])

    for nombre, tecnica in modelos.items():
        kf = KFold(n_splits= tecnica.CV)
        for i, (train_index, test_index) in enumerate(kf.split(predictores, target)):
            X_train, t_train = predictores.iloc[train_index, :], target.iloc[train_index]
            X_test, t_test = predictores.iloc[test_index, :], target.iloc[test_index]

            tecnica.entrenar(X_train, t_train)
            prediccion = tecnica.predecir(X_test)

            v = error.calculo(metricas, t_test, prediccion)
            v = np.append(v, [i])
            df1.loc[n]= v
            df2.loc[n]= [nombre]

            n+=1
    return pd.concat([df1, df2], axis= 1, join= "inner")


#Función encargada de crear un DataFrame con los valores de las métricas de los modelos.
#Este DF posee una columna con cada test, una indicando el modelo para ese test, y una última columna con la iteración.
def crearDF(modelos, metricas):
    """
    Crea un DataFrame con los valores de las métricas de los modelos. Este DF posee una columna por cada métrica, más una columna con el nombre del modelo y otra con la iteración.
    """
    indice= 0
    columnas = list(metricas.keys())
    columnas.append('Iteracion')  
    df1 = pd.DataFrame(columns= columnas)
    df2 = pd.DataFrame(columns= ['Tecnica'])

    #Se van concatenando, para cada modelo ->
    for nombre, modelo in modelos.items():
        n_test = modelo.CV
        #en cada una de los iteraciones ->
        for iteracion in range(n_test):
            error = []
            #cada una de las métricas ->
            for metrica in metricas.values():
                error = np.concatenate((error, [modelo.scores['test_'+ metrica][iteracion]]))
            df1.loc[indice] = np.concatenate((error, [iteracion]))
            df2.loc[indice] = nombre

            indice+= 1

    return pd.concat([df1, df2], axis=1, join="inner")