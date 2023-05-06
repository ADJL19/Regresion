#Importación de las librerías empleadas.
import json
import pandas as pd

import gph
import fnc
from crearModelos import definirModelos

from timeit import timeit

def main():
    #Cargamos el archivo JSON donde se encuentra la configuración deseada.
    settings = json.load(open("settings.json", "r"))

    #Importamos los valores de predictores y target según la configuración.
    [target, predictores] = fnc.importacionDatos(settings)

    #Podemos representar los datos importados en una ScatterPlotMatrix.
    if settings["representacion"]["ScatterMatrix"]: gph.representarDatos(pd.concat([predictores, target], axis=1, join="inner"))

    #Contruimos los modelos, que son almacenados en un diccionario.
    modelos= definirModelos(settings, predictores.shape[1])

    #Se obtienen las métricas a evaluar y el KFold que se empleará en la validación cruzada de la configuración
    metricas= settings['entrenamiento']['metrics']
    CV= settings['entrenamiento']['CV']

    #Se realiza la validación cruzada en función del método seleccionado.
    data= fnc.validacionCruzada(modelos, predictores, target, metricas, CV)
    data= fnc.validacionCruzadaMultiModelo(modelos, predictores, target, metricas, CV)
    data= fnc.validacionCruzadaMultiKFold(modelos, predictores, target, metricas, CV)

    gph.variasBoxplot(data)

if __name__ == '__main__':
    main()