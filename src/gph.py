import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def matrizCorrelacion(data):
    """
    Crear una gráfica de correlación.

    Parámetros:
    ----------
    data: DataFrame con las variables sobre las que calcular la matriz de correlación.

    Devuelve:
    ----------
    d: DataFrame con la matriz de correlación.
    """
    n= np.corrcoef(data.T)
    etiquetas= data.columns
    sns.heatmap(n, vmin=-1, vmax=1, linewidths=1, cmap= 'BrBG',
                xticklabels=etiquetas, yticklabels=etiquetas, annot= True)
    plt.xticks(rotation= 45)
    plt.show()

    d = pd.DataFrame(n, index= etiquetas, columns= etiquetas)
    return d

#Función encargada de representar datos en un Scatterplot matricial
def representarDatos(data):
    """
    Representa los datos en un Scatterplot matricial.

    Parámetros:
    ----------
    data: DataFrame con las variables a representar.
    """
    sns.set_theme(style="ticks")
    sns.pairplot(data)
    plt.xticks(rotation= 45)
    plt.yticks(rotation= 45)
    plt.show()


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
    for metrica in metricas:
        for m in metrica:
            boxplot(data, m)


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
    for metrica in metricas:
        for m in metrica:
            scatter(data, m)