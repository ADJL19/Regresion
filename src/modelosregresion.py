from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate

class modelo():
    def __init__(self, modelo):
        self.__prediccion = []
        self.__scores = []
        self.__modelo = modelo
        self.__scoring = ['max_error', 'neg_mean_absolute_error', 'neg_root_mean_squared_error', 'neg_median_absolute_error', 'r2']
        self.__CV = 10

    def entrenarModelo(self, train, t_train):
        self.__modelo = self.__modelo.fit(train, t_train)

    def predecir(self, test):
        self.__prediccion = self.__modelo.predict(test)
        return self.__prediccion
    
    @property
    def prediccion(self):
        return self.__prediccion
    
    @property
    def hiperparametros(self):
        return self.__modelo.get_params(False)
    
    def validacionCruzada(self, predictores, etiquetas):
        self.__scores = cross_validate(self.__modelo, predictores, etiquetas, scoring= self.__scoring, cv= self.__CV)
        return self.__scores

def main():
    juan = modelo(LinearRegression())
    predictores = [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]]
    etiquetas = [[1], [2], [3], [4], [5]]

    juan.entrenarModelo(predictores, etiquetas)
    print(juan.predecir(predictores))
    print(juan.hiperparametros)

if __name__ == "__main__":
    main()