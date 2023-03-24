#Importación de las librerías empleadas
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Función encargada de importar los datos desde el .csv, separándolos ya 
def importacionDatos(path):
    data = pd.read_csv(path)
    
    #Eliminamos aquellas filas en las que la intensidad sean 0A
    data = data[data['I'] > 0]

    #Realizamos el reseteo dell DF para obtener sus índices ordenados desde 0 hasta N,
    #en lugar de desde 0 hasta el máximo original, pero con tan solo N índices.
    data = pd.concat([data], ignore_index= True)
    target = data.Enerxia.values

    #Normalizamos los datos de entrada a los mdelos
    predictores = normalizar(data.loc[:, ["VelocidadeDoVentoA10m", "RefachoA10m", "Presion", "Velocidade"]].values)

    return target, predictores

def normalizar(data):
    scaler = StandardScaler()
    scaler.fit(data)
    return scaler.transform(data)

#Función que realiza la validación cruzada de los distintos modelos.
def validacionCruzada(modelos, predictores, target, metricas):
    #Se recorre el diccionario donde se encuentran los modelos.
    for nombre, tecnica in modelos.items():
        print(f"Para el modelo {nombre} con parámetros {tecnica.parametros}:")
        #Para cada modelo se realiza la validación cruzada
        scores = tecnica.validacionCruzada(predictores, target)
        #Se recorre cada una de las métricas evaluadas
        for test, valor in metricas.items():
            #Indicando el nombre del test y el velor medio de este
            print(f"El {test} vale {np.mean(scores['test_' + valor])}")
        print("")

#Función encargada de crear un DataFrame con los valores de las métricas de los modelos.
#Este DF posee una columna con cada test, una indicando el modelo para ese test, y una última columna con la iteración.
def crearDF(modelos, metricas):
    n_test = list(modelos.values())[0].CV
    n, t, error = 0, 0, []

    nombre = list(metricas.keys())
    nombre.append('Iteracion')
    nombre.append('Tecnica')

    print()
    
    df = pd.DataFrame(columns= nombre)

    #Se van concatenando, para cada modelo ->
    for modelo in modelos.values():
        #cada una de los iteraciones ->
        for i in range(n_test):
            #cada una de las métricas ->
            for score in metricas.values(): 
                error = np.concatenate((error, [modelo.scores['test_'+ score][i]]))
            df.loc[n] = np.concatenate((error, [i], [t]))
            n += 1
            error = []
        t += 1
    return df

#Funcion encargada del dibujado de una gráfica de tipo 'boxplot'.
def boxplot(modelos, data, metrica):
    sns.set_theme(style="ticks")
    _, ax = plt.subplots(figsize=(5, 5))

    #Agrupando las técnicas, dibujará el diagrama de caja en función de la métrica seleccionada.
    sns.boxplot(x=data.Tecnica, y=metrica, data=data)

    ax.xaxis.grid(True)
    ax.set(ylabel= metrica, xlabel=list(modelos.keys()))
    sns.despine(trim=True, left=True)
    plt.show()

def variasBoxplot(modelos, data, *graficas):
    for i in graficas:
        boxplot(modelos, data, i)

#Función encargada de dibujar un gráfico de dispersión
def scatter(modelos, data, metrica):
    sns.set_theme()
    _, ax = plt.subplots(figsize=(5, 5))

    #Agrupando por iteracion, dibuja para cada modelo la métrica seleccionada. Le da un estilo y color en función de la técnica.
    sns.scatterplot(data=data, x=data.Iteracion, y=metrica, hue=data.Tecnica, style=data.Tecnica)

    ax.xaxis.grid(True)
    ax.set(ylabel= metrica, xlabel=list(modelos.keys()))
    sns.despine(trim=True, left=True)
    plt.show()

def variasScatter(modelos, data, *graficas):
    for i in graficas:
        scatter(modelos, data, i)