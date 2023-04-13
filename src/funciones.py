#Importación de las librerías empleadas
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Función encargada de importar los datos desde el .csv.
def importacionDatos(path, normalizar= True, reduccion= None):
    #Leemos el archivo donde se almacenan los datos
    data = pd.read_csv(path)

    #Eliminamos los datos espurios
    data = data[data['Enerxia'] > 0]
    # data = data[data['I'] > 0]
    # data = data[data['Velocidade'] > 3.5]
    data = data[data['Velocidade'] < 14]

    #Realizamos el reseteo del DF para obtener sus índices ordenados desde 0 hasta N,
    #en lugar de desde 0 hasta el máximo original, pero con tan solo N índices.
    data = pd.concat([data], ignore_index= True)

    #Se aleatorizan los datos
    data = data.sample(n = len(data))

    #Se dividen los datos en target y predictores
    target = data.Enerxia
    data = data.drop(columns= ['Time', 'Enerxia', 'V', 'I', 'W', 'VAr', 'Wh_e'])

    #Se normalizan los datos si así de indica.
    if normalizar: data = normalizacion(data)

    #Se realiza la reducción introducida
    if reduccion!= None:
        reduccion.fit(data) 
        predictores= reduccion.transform(data)
    else:
        predictores = data[:, [0, 17, 3, 4]]

    return target, predictores

def normalizacion(data):
    scaler = StandardScaler()
    scaler.fit(data)
    return scaler.transform(data)

#Función que realiza la validación cruzada de los distintos modelos.
def validacionCruzada(modelos, predictores, target, metricas):
    #Se recorre el diccionario donde se encuentran los modelos.
    for nombre, tecnica in modelos.items():
        print(f"Para el modelo {nombre} con parámetros {tecnica.parametros}:")
        #Para cada modelo se realiza la validación cruzada
        scores = tecnica.validacionCruzada(predictores, target, list(metricas.values()))
        #Se recorre cada una de las métricas evaluadas
        for test, valor in metricas.items():
            # Indicando el nombre del test y el valor medio de este
            print(f"El {test} vale {np.mean(scores['test_' + valor])}")
        print("")

    return crearDF(modelos, metricas)

#Función encargada de crear un DataFrame con los valores de las métricas de los modelos.
#Este DF posee una columna con cada test, una indicando el modelo para ese test, y una última columna con la iteración.
def crearDF(modelos, metricas):
    n, error = 0, []

    columnas = list(metricas.keys())
    columnas.append('Iteracion')  
    df1 = pd.DataFrame(columns= columnas)
    df2 = pd.DataFrame(columns= ['Tecnica'])

    #Se van concatenando, para cada modelo ->
    for nombre, modelo in modelos.items():
        n_test = modelo.CV
        #cada una de los iteraciones ->
        for i in range(n_test):
            #cada una de las métricas ->
            for score in metricas.values(): 
                error = np.concatenate((error, [modelo.scores['test_'+ score][i]]))
            df1.loc[n] = np.concatenate((error, [i]))
            df2.loc[n] = nombre
            n += 1
            error = []

    return pd.concat([df1, df2], axis=1, join="inner")

#Funcion encargada del dibujado de una gráfica de tipo 'boxplot'.
def boxplot(data, metrica):
    sns.set_theme(style= "ticks")
    _, ax = plt.subplots(figsize= (10, 5))

    #Agrupando las técnicas, dibujará el diagrama de caja en función de la métrica seleccionada.
    sns.boxplot(data= data, y= metrica, x= "Tecnica")

    ax.xaxis.grid(True)
    ax.set(ylabel= metrica)
    sns.despine(trim= True, left= True)
    plt.show()

def variasBoxplot(data, *metricas):
    for i in metricas:
        boxplot(data, i)

#Función encargada de dibujar un gráfico de dispersión
def scatter(data, metrica):
    sns.set_theme()
    _, ax = plt.subplots(figsize=(5, 5))

    #Agrupando por iteracion, dibuja para cada modelo la métrica seleccionada. Le da un estilo y color en función de la técnica.
    sns.scatterplot(data=data, x= "Iteracion", y=metrica, hue="Tecnica", style=data.Tecnica)

    ax.xaxis.grid(True)
    ax.set(ylabel= metrica)
    sns.despine(trim=True, left=True)
    plt.show()

def variasScatter(data, *metricas):
    for i in metricas:
        scatter(data, i)
