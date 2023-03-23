import datos

from sklearn.linear_model import LinearRegression
from modelosregresion import modelo

path = "./data/output.csv"
[etiquetas, predictores] = datos.importacionDatos(path)

scoring = ['max_error', 'neg_mean_absolute_error', 'neg_root_mean_squared_error', 'neg_median_absolute_error', 'r2']
errores = ['Error máximo', 'MAE', 'MSE', 'MAE', 'Coeficiente de correlación']

miModelo1 = modelo(LinearRegression())


# for valor, test in zip(scoring, errores):
#     print(f"El {test} vale {np.mean(scores['test_' + valor])}")