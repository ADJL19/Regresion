import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import datos
import metricaserror as error

from sklearn import preprocessing, neighbors
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline

from modelosregresion import modelo

scoring = ['max_error', 'neg_mean_absolute_error', 'neg_root_mean_squared_error', 'neg_median_absolute_error', 'r2']
errores = ['Error máximo', 'MAE', 'MSE', 'MAE', 'Coeficiente de correlación']

# modelos = []
# modelos.append(('LinR', LinearRegression()))
# n_neighbors = 5
# modelos.append(('KNN', neighbors.KNeighborsRegressor(n_neighbors)))

path = "./data/output.csv"

[etiquetas, predictores] = datos.importacionDatos(path)

miModelo = modelo()
miModelo.entrenarModelo(predictores, etiquetas)
y_pred = miModelo.predecir(predictores)
print(error.MSE(etiquetas, y_pred))

# CV = 10
# for (nombre, modelo) in modelos:
#     modelo = make_pipeline(preprocessing.StandardScaler(), modelo)
# scores = cross_validate(miModelo, predictores, etiquetas, cv=CV, scoring=scoring)
#     for i, test in enumerate(scoring):
#         print(f"La media de {nombre} para [{errores[i]}] es: {np.mean(scores['test_' + test])}")
#     print("")

# plt.style.use('_mpl-gallery')
# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# ax.plot(predictores.x, predictores.y, etiquetas, 'ro')

# U = np.arange(10, 21, 0.5)
# V = np.arange(10, 21, 0.5)
# hola = scores['estimator']
# hola.predict([U, V])

# ax.plot_surface(U, V, Z)
# plt.show()