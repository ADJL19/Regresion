#Importación de las librerías empleadas.
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import gph
import fnc
from crearModelos import definirModelos

from timeit import timeit

def main():
    #Cargamos el archivo JSON donde se encuentra la configuración deseada.
    with open("config.json", "r") as config_file: config = json.load(config_file)

    # #Importamos los valores de predictores y target según la configuración.
    [target, predictores] = fnc.importacionDatos(config)

    #Podemos representar los datos importados, bien sea para comprobar las variables utilizadas, la relación entre ellas o etc.
    if config["representacion"]["ScatterMatrix"]: gph.representarDatos(pd.concat([predictores, target], axis=1, join="inner"))

    modelos= definirModelos(config, predictores.shape[1])
    metricas= config['entrenamiento']['metrics']
    CV= config['entrenamiento']['CV']

    data= fnc.validacionCruzada(modelos, predictores, target, metricas, CV)
    print(data)
    data= fnc.validacionCruzadaMultiModelo(modelos, predictores, target, metricas, CV)
    print(data)
    # data= fnc.validacionCruzadaMultiKFold(modelos, predictores, target, metricas, CV)
    # print(data)

    gph.variasBoxplot(data)

if __name__ == '__main__':
    main()