import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import datos
import metricaserror as error

from sklearn import preprocessing, neighbors
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline

scoring = ['max_error', 'neg_mean_absolute_error', 'neg_root_mean_squared_error', 'neg_median_absolute_error', 'r2']
modelos = []
modelos.append(('LinR', LinearRegression()))
n_neighbors = 5
modelos.append(('KNN', neighbors.KNeighborsRegressor(n_neighbors)))

path = "C:/Users/adzl/Desktop/AA/Traballo/Datos/Datos/output.csv"
[etiquetas, predictores] = datos.importacionDatos(path)

test_sc = []
CV = 10

for (nombre, modelo) in modelos:
    modelo = make_pipeline(preprocessing.StandardScaler(), modelo)
    scores = cross_validate(modelo, predictores, etiquetas, cv=CV, scoring=scoring, return_estimator= True)
    print(f"La media de {nombre}(MAE) es: {np.mean(scores['test_neg_mean_absolute_error'])}")

plt.style.use('_mpl-gallery')
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.plot(predictores.x, predictores.y, etiquetas, 'ro')

U = np.arange(10, 21, 0.5)
V = np.arange(10, 21, 0.5)
hola = scores['estimator']
hola.predict([U, V])

# ax.plot_surface(U, V, Z)
# plt.show()