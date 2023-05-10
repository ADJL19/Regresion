#Importación de las librerías empleadas
import os
import numpy as np
import pandas as pd

import gph
import metricaserror as error

from threading import Thread
from multiprocessing import Process

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FastICA
from sklearn.model_selection import KFold

def importacionDatos(config):
    """Importa los datos de archivos Excel o CSV.

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
    if config["preprocesamiento"]["aleatorizar"]: data = data.sample(n = len(data))

    #Se dividen los datos en target y predictores
    target = data.Energy

    #Se eliminan del DF las variables no útiles.
    data = data.drop(columns= config["preprocesamiento"]["variablesEspurias"])

    if config["representacion"]["MatrizCorrelacion"]: gph.matrizCorrelacion(pd.concat([data, target], axis=1, join="inner"))

    #Se normalizan los datos si así de indica.
    if config["preprocesamiento"]["normalizar"]: data= pd.DataFrame(StandardScaler().fit_transform(data), columns= data.columns)

    #Se realiza la reducción introducida:
    if config["preprocesamiento"]["reduccion"]["PCA"]:
        reduccion= PCA(n_components= config["preprocesamiento"]["PCA"]["n_components"])
        predictores= reduccion.fit_transform(data)

        variablesUsadas= vReduccion(reduccion, data)
        predictores = pd.DataFrame(predictores, columns= variablesUsadas)
    elif config["preprocesamiento"]["reduccion"]["ICA"]:
        reduccion= FastICA(n_components= config["preprocesamiento"]["ICA"]["n_components"])
        predictores= reduccion.fit_transform(data)

        variablesUsadas= vReduccion(reduccion, data)
        predictores = pd.DataFrame(predictores, columns= variablesUsadas)
    else:
        predictores = data.loc[:, config["preprocesamiento"]["variablesExplicativas"]]

    return target, predictores

def vReduccion(model, data):
    nComp= len(model.components_)
    mejorExplicacion = [np.abs(model.components_[comp]).argmax() for comp in range(nComp)]
    variables = data.columns
    return [variables[mejorExplicacion[comp]] for comp in range(nComp)]

def exportacionExcel(settings, DF, modelos):
    for modelo, tf in settings["modelos"].items():
        if tf:
            archivo = settings["exportacion"]["ruta"] + os.sep + modelo + '.xlsx'
            writer = pd.ExcelWriter(archivo)
            for nombre in modelos.keys(): 
                if modelo in nombre:
                    DF[DF.Tecnica== nombre].to_excel(writer, sheet_name= nombre, index= True)
            writer.close()
    print("Exportación a .xlsx realizada.")

def crearDF(metricas):
    col = metricas.copy()
    col.append('Iteracion')
    col.append('Tecnica')
    return pd.DataFrame(columns= col)

def validacionCruzada(modelos, predictores, target, metricas, CV):
    """Realiza la validación cruzada de los modelos.

    Parámetros:
    ----------
    modelos: Diccionario con los modelos sobre los que realizar la validación cruzada.
    predictores: Array-like numérico con las variables explicativas.
    target: Array-like numérico con los resultados reales de la predicción.
    metricas: Array-like de texto con las métricas de error a evaluar.

    Devuelve:
    ----------
    d: Dataframe con el resultado de las métricas.
    """

    DF = crearDF(metricas)
    kf = KFold(n_splits= CV)
    for nombre, tecnica in modelos.items():
        DF= pd.concat([DF, validacionCruzadaModelo(nombre, tecnica, predictores, target, metricas, kf)], axis=0, join='inner')
    return DF

def validacionCruzadaModelo(nombre, tecnica, predictores, target, metricas, kf):
    """Realiza la validación cruzada K-Fold para un modelo.

    Parámetros:
    ----------
    nombre : String 
        Nombre que se le da al modelo.

    tecnica: Objeto 
        Modelo.

    predictores : Array-like de (n_muestras) o matrix-like de (n_muestras x n_variables) 
        Valor de la(s) variable(s) explicativa(s).

    target : Array-like de (n_muestras)
        Valor real de las predicciones.

    metricas : Array-like de (n_metricas)
        Métricas que se desean calcular.

    Devuelve:
    ----------
    d: Dataframe 
        Valor de las métricas.
    """

    DF= crearDF(metricas)
    for i, (train_index, test_index) in enumerate(kf.split(predictores, target)):
        v= validacionCruzadaKFold(nombre, tecnica, predictores, target, metricas, train_index, test_index, i)
        DF= pd.concat([DF, v], axis= 0, join= 'inner', ignore_index= True)
    return DF

def validacionCruzadaKFold(nombre, tecnica, predictores, target, metricas, train_index, test_index, i):
    DF= crearDF(metricas)
    X_train, t_train = predictores.iloc[train_index, :], target.iloc[train_index]
    X_test, t_test = predictores.iloc[test_index, :], target.iloc[test_index]
    
    tecnica.entrenar(X_train, t_train)
    prediccion = tecnica.predecir(X_test)

    v = np.append(error.calculo(metricas, t_test, prediccion), [i])
    DF.loc[0]= np.append(v, [nombre])
    DF[DF.columns[:-1]]= DF[DF.columns[:-1]].astype('float')
    return DF

class MiHilo(Thread):
    """Clase creada mediante la herencia de Thread. Se le añade una propiedad para acceder al resultado del hilo.
    
    Parámetros:
    ----------
    target : Función
        Función que se implementará en un hilo.

    args : Tupla
        Argumentos de la función objetivo

    Devuelve:
    ----------
    result : 
        Resultado de la función target
    """
    def __init__(self, target, args):
        super().__init__(target= target, args= args)

    def run(self):
        self.result= self._target(*self._args)

    def result(self):
        return self.result
    
class MiProceso(Process):
    def __init__(self, target, args):
        super().__init__(target= target, args= args)

    def run(self):
        self.result= self._target(*self._args)

    def result(self):
        return self.result

def validacionCruzadaMultiModelo(modelos, predictores, target, metricas, CV):
    """Realiza la validación cruzada multihilo para cada modelo. De esta manera, en cada hilo se realiza toda la validación cruzada de un modelo.
    """
    hilos = []
    kf= KFold(n_splits= CV)
    DF= crearDF(metricas)

    for nombre, tecnica in modelos.items():
        hilo = MiHilo(target= validacionCruzadaModelo, args=(nombre, tecnica, predictores, target, metricas, kf))
        hilo.start()
        hilos.append(hilo)
    for hilo in hilos:
        hilo.join()
    for hilo in hilos:
        df = hilo.result
        DF = pd.concat([DF, df], axis=0, join= 'inner')
    return pd.concat([DF], ignore_index= True)

def validacionCruzadaMultiKFold(modelos, predictores, target, metricas, CV):
    hilos = []
    DF= crearDF(metricas)

    kf = KFold(n_splits= CV)
    for nombre, tecnica in modelos.items():
        for i, (train_index, test_index) in enumerate(kf.split(predictores, target)):
            hilo = MiHilo(target= validacionCruzadaKFold, args= (nombre, tecnica, predictores, target, metricas, train_index, test_index, i))
            hilo.start()
            hilos.append(hilo)
    for hilo in hilos:
        hilo.join()
    for hilo in hilos:
        df = hilo.result
        DF = pd.concat([DF, df], axis=0, join= 'inner')

    return pd.concat([DF], ignore_index= True)