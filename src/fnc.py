#Importación de las librerías empleadas
import numpy as np
import pandas as pd

import gph
import metricaserror as error
from threading import Thread

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FastICA
from sklearn.model_selection import KFold

def importacionDatos(config):
    """
    Importa los datos de archivos Excel o CSV.

    Parámetros:
    ----------
    config: JSON donde se establece la ruta del archivo, su extensión y el separador en caso de ser archivos CSV. Además, en este archivo, se establece si se realizará la normalización de los datos o si se aplicará algún método de reducción de dimensionalidad.

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
    if config["fichero_datos"]["normalizar"]: data= pd.DataFrame(StandardScaler().fit_transform(data), columns= data.columns)

    #Se realiza la reducción introducida:
    if config["reduccion"]["PCA"]:
        reduccion= PCA(n_components= config["PCA"]["n_components"])
        predictores= reduccion.fit_transform(data)

        variablesUsadas= vReduccion(reduccion, data)
        predictores = pd.DataFrame(predictores, columns= variablesUsadas)
    elif config["reduccion"]["ICA"]:
        reduccion= FastICA(n_components= config["ICA"]["n_components"])
        predictores= reduccion.fit_transform(data)

        variablesUsadas= vReduccion(reduccion, data)
        predictores = pd.DataFrame(predictores, columns= variablesUsadas)
    else:
        predictores = data.loc[:, ['Wind_V', 'Wind_V_10m', 'SD_Wind_V_10m', 'SD_Wind_D_10m']]

    return target, predictores

def vReduccion(model, data):
    nComp= len(model.components_)
    mejorExplicacion = [np.abs(model.components_[comp]).argmax() for comp in range(nComp)]
    variables = data.columns
    return [variables[mejorExplicacion[comp]] for comp in range(nComp)]

def validacionCruzada(modelos, predictores, target, metricas):
    """
    Realiza la validación cruzada de los modelos.

    Parámetros:
    ----------
    modelos: Diccionario con los modelos sobre los que realizar la validación cruzada.
    predictores: Array-like numérico con las variables explicativas.
    target: Array-like numérico con los reultados reales de la predicción.
    metricas: Array-like de texto con las métricas de error a evaluar.

    Devuelve:
    ----------
    d: Dataframe con el resultado de las métricas.
    """

    col = metricas.copy()
    col.append('Iteracion')
    col.append('Tecnica')

    df1 = pd.DataFrame(columns= col)
    for nombre, tecnica in modelos.items():
        df1= validacionCruzadaSimple(nombre, tecnica, predictores, target, metricas, df1)

    df1[col[:-1]]= df1[col[:-1]].astype('float32')

    return df1

def validacionCruzadaSimple(nombre, tecnica, predictores, target, metricas, df1):
    kf = KFold(n_splits= tecnica.CV)
    for i, (train_index, test_index) in enumerate(kf.split(predictores, target)):
        X_train, t_train = predictores.iloc[train_index, :], target.iloc[train_index]
        X_test, t_test = predictores.iloc[test_index, :], target.iloc[test_index]

        tecnica.entrenar(X_train, t_train)
        prediccion = tecnica.predecir(X_test)

        v = error.calculo(metricas, t_test, prediccion)
        v= np.append(v, [i])
        v = pd.DataFrame([np.append(v, [nombre])], columns= df1.columns)

        df1= pd.concat([df1, v], axis= 0, join= 'inner', ignore_index= True)
    return df1

class MiHilo(Thread):
    def __init__(self, nombre, tecnica, predictores, target, metricas, **kwargs):
        super().__init__(**kwargs)
        self.nombre = nombre
        self.tecnica = tecnica
        self.predictores = predictores
        self.target = target
        self.metricas = metricas

    def run(self):
        validacionCruzadaSimple(self.nombre, self.tecnica, self.predictores, self.target, self.metricas)

def validacionCruzadaMulti(modelos, predictores, target, metricas):
    hilos = []
    for nombre, tecnica in modelos.items():
        hilo = MiHilo(nombre, tecnica, predictores, target, metricas)
        hilo.start()
        hilos.append(hilo)
    for hilo in hilos:
        hilo.join()