import numpy as np
import pandas as pd
import scipy.stats as stats

import matplotlib.pyplot as plt
import seaborn as sns

def importacionDatos(path):
    data = pd.read_csv(path)

    etiquetas = data.iloc[:, -1]
    datos = data.iloc[:, [0, 1]].values

    return etiquetas, datos

def validacionCruzada(modelos, predictores, etiquetas, metricas):
    for nombre, tecnica in modelos.items():
        print(f"Para el modelo {nombre} con parámetros {tecnica.hiperparametros}:")
        scores = tecnica.validacionCruzada(predictores, etiquetas)
        for test, valor in metricas.items():
            print(f"El {test} vale {np.mean(scores['test_' + valor])}")
        print("")

def crearDF(modelos, metricas):
    error = []
    tecnica = [[i] * list(modelos.values())[0].CV for i in range(len(modelos))]
    iteracion = [i for j in range(len(modelos)) for i in range(list(modelos.values())[0].CV)]

    for nombre, modelo in modelos.items():
        for score in metricas.values():  
            error = np.concatenate((error, modelo.scores['test_'+ score]))

    error = error.reshape((5, -1), order= 'A').T
    tecnica = np.reshape(tecnica, (-1, 1))
    iteracion = np.reshape(iteracion, (-1, 1))
    datos = np.hstack((error, tecnica, iteracion))

    del error
    del tecnica

    return pd.DataFrame(datos, columns= ['MeanAE', 'R2', 'MAX_ERROR', 'MSE', 'MedianAE', 'Tecnica', 'Iteracion'])

def boxplot(data, metrica):
    sns.set_theme(style="ticks")

    _, ax = plt.subplots(figsize=(7, 6))

    sns.boxplot(x=data.Tecnica, y=metrica, data=data)

    ax.xaxis.grid(True)
    ax.set(ylabel="")
    sns.despine(trim=True, left=True)
    plt.show()

def scatter(data, metrica):
    sns.scatterplot(data=data, x=data.Iteracion, y=metrica, hue=data.Tecnica, style=data.Tecnica)

def contrasteHipotesis(*samples, alpha=0.05):
    for sample in samples:
        print(sample)

    [_, pval1] = stats.kruskal()
    [_, pval2] = stats.f_oneway()
    if pval1 <= alpha:
        print('Se rechaza la hipótesis: los modelos son diferentes.')

    else: print('Se acepta la hipótesis: los modelos son iguales.')