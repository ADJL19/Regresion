from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate

class modelo():
    def __init__(self, modelo= LinearRegression()):
        self.__modelo = modelo
        self.__prediccion = []
        self.__scores = []
        self.__scoring = ['max_error', 'neg_mean_absolute_error', 'neg_root_mean_squared_error', 'neg_median_absolute_error', 'r2']
        self.__CV = 5

    @property
    def hiperparametros(self):
        return self.__modelo.get_params(False)

    def entrenarModelo(self, train, t_train):
        self.__modelo = self.__modelo.fit(train, t_train)

    def predecir(self, test):
        self.__prediccion = self.__modelo.predict(test)
        return self.__prediccion
    
    @property
    def prediccion(self):
        return self.__prediccion
    
    def validacionCruzada(self, predictores, etiquetas):
        self.__scores = cross_validate(self.__modelo, predictores, etiquetas, scoring= self.__scoring, cv= self.__CV)
        return self.__scores
    
    @property
    def scores(self):
        return self.__scores
    
    @property
    def CV(self):
        return self.__CV