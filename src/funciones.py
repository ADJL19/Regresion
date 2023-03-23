import numpy as np
import pandas as pd

def crearDF(modelos, metrica):
    error = []
    tecnica = [[i] * list(modelos.values())[0].CV for i in range(len(modelos))]

    for nombre, modelo in modelos.items():
        for score in metrica:  
            error = np.concatenate((error, modelo.scores['test_'+ score]))

    error = error.reshape((5, -1), order= 'A').T
    tecnica = np.reshape(tecnica, (-1, 1))
    datos = np.hstack((error, tecnica))
    return pd.DataFrame(datos, columns= ['MAE', 'R2', 'MAX_ERROR', 'MSE', 'RMSE', 'Tecnica'])