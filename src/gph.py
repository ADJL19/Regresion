import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def matrizCorrelacion(data):
    MC= np.corrcoef(data.T)
    etiquetas= data.columns
    sns.heatmap(MC, vmin=-1, vmax=1, linewidths=1, cmap= 'BrBG',
            xticklabels=etiquetas, yticklabels=etiquetas, annot= True)
    plt.xticks(rotation= 90)
    plt.show()

#Función encargada de representar datos en un Scatterplot matricial
def representarDatos(data):
    sns.set_theme(style="ticks")
    sns.pairplot(data)
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
        boxplot(data, metrica)


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
        scatter(data, metrica)