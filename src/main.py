#Importación de las librerías empleadas.
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import gph
import fnc
from crearModelos import definirModelos

def main():
    #Cargamos el archivo JSON donde se encuentra la configuración deseada.
    with open("hiperparametros.json", "r") as config_file: config = json.load(config_file)

    # #Importamos los valores de predicores y target segun la configuración.
    [target, predictores] = fnc.importacionDatos(config)

    # #Podemos representar los datos importados, bien sea para comprobar las variables utilizadas, la relación entre ellas o etc.
    # gph.representarDatos(pd.concat([predictores, target], axis=1, join="inner"))

    modelos= definirModelos(config, predictores.shape[1])
    data= fnc.validacionCruzada(modelos, predictores, target, metricas= config["entrenamiento"]["metrics"])

    gph.variasBoxplot(data, "RMSE", "MeanAE")

if __name__ == '__main__':
    main()