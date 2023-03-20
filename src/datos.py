import pandas as pd
import numpy as np


def importacionDatos(path):
    data = pd.read_csv(path)

    etiquetas = data.iloc[:, -1]
    datos = data.iloc[:, [0, 1]]

    return etiquetas, datos
