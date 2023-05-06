#Importación de las librerías empleadas
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate

class model():
    #Inicialización de la clase. Por defecto, se establece un modelo de regresión lineal.
    def __init__(self, modelo= LinearRegression()):
        self.__modelo = modelo
        self.__prediccion = []
        self.__scores = []

    #Propiedad para obtener el valor de los parámetros del modelo
    @property
    def parametros(self):
        return self.__modelo.get_params(False)

    #Método que realiza el entrenamiento del modelo.
    def entrenar(self, train, t_train):
        self.__modelo = self.__modelo.fit(train, t_train)

    #Método que realiza la predicción para el modelo
    def predecir(self, test):
        self.__prediccion = self.__modelo.predict(test)
        return self.__prediccion

    #Método que realiza la validación cruzada del modelo. Devuelve las métricas pedidas
    def validacionCruzada(self, predictores, etiquetas, scoring= ['max_error', 'neg_mean_absolute_error', 'neg_root_mean_squared_error', 'neg_median_absolute_error', 'r2']):
        self.__scores = cross_validate(self.__modelo, predictores, etiquetas, scoring= scoring, cv= self.__CV)
        return self.__scores
    
    #Propiedad para obtener, en cualquier momento, la última predicción realizada.
    @property
    def prediccion(self):
        return self.__prediccion

    #Propiedad para obtener, en cualquier momento, el valor de las métricas obtenida en la validación cruzada.
    @property
    def scores(self):
        return self.__scores