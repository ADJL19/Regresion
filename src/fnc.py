#Importación de las librerías empleadas
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd

import gph
import metricaserror as error

#Función encargada de importar los datos desde el .csv.
def importacionDatos(path, normalizar= True, reduccion= None):
    #Leemos el archivo donde se almacenan los datos
    data = pd.read_csv(path)

    # #Eliminamos los datos espurios
    # data = data[data['Enerxia'] > 0]
    # data = data[data['Enerxia'] < 1500]
    # data = data[data['Velocidade'] < 14]
    # data = data[data['I'] > 0]
    # data = data[data['Velocidade'] > 3.5]

    #Realizamos el reseteo del DF para obtener sus índices ordenados desde 0 hasta N,
    #en lugar de desde 0 hasta el máximo original, pero con tan solo N índices.
    data = pd.concat([data], ignore_index= True)

    #Se aleatorizan los datos
    data = data.sample(n = len(data))

    #Se dividen los datos en target y predictores
    target = data.Energy

    data = data.drop(columns= ['Time', 'Energy', 'V', 'I', 'W', 'VAr', 'Wh_e'])
    
    # gph.matrizCorrelacion(pd.concat([data, target], axis=1, join="inner"))

    #Se normalizan los datos si así de indica.
    if normalizar: 
        data= pd.DataFrame(normalizacion(data), columns= data.columns)

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