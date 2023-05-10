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

    data= fnc.validacionCruzada(modelos, predictores, target, metricas, CV)

    # Se realiza la validación cruzada en función del método seleccionado.
    num= 10
    t1 = timeit(lambda: fnc.validacionCruzada(modelos, predictores, target, metricas, CV), number= num, globals=globals())
    print(f"Ejecución simple realizada en: {t1/60:.5f} minutos.")
    t2= timeit(lambda: fnc.validacionCruzadaMultiModelo(modelos, predictores, target, metricas, CV), number= num, globals=globals())
    print(f"Ejecución con un modelo por hilo realizada en: {t2/60:.5f} minutos.")
    t3= timeit(lambda: fnc.validacionCruzadaMultiKFold(modelos, predictores, target, metricas, CV), number= num, globals=globals())
    print(f"Ejecución con un entrenamiento/predicción por hilo realizada en: {t3/60:.5f}.")

    if settings["entrenamiento"]["mostrar"]["texto"]: fnc.mostrarComparacion(data, settings["entrenamiento"]["comparacion"])
    if settings["entrenamiento"]["mostrar"]["grafica"]: gph.variasBoxplot(data, settings["entrenamiento"]["comparacion"])
    if settings["exportacion"]["excel"]: fnc.exportacionExcel(settings, data, modelos)



if __name__ == '__main__':
    main()