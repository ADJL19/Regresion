from sklearn.linear_model import LinearRegression


class regresionlineal:
    def __init__(self, fit_intercept= True, n_jobs= None, positive= False):
        self.__fit_intercept = fit_intercept
        self.__n_jobs = n_jobs
        self.__positive = positive

    def __fit_intercept(self):
        return self.__fit_intercept
    
    def __n_jobs(self):
        return self.__n_jobs
    
    def __positive(self):
        return self.__positive
    
    @property
    def hiperparametros(self):
        return self.__fit_intercept, self.__n_jobs, self.__positive

    def _crearModelo(self):
        return LinearRegression(fit_intercept= self.__fit_intercept, n_jobs= self.__n_jobs, positive= self.__positive)
    
class modelo(regresionlineal):
    def __init__(self):
        super().__init__()
        self.__prediccion = []

    def entrenarModelo(self, train, t_train):
        self.__modelo = self._crearModelo()
        self.__modelo = self.__modelo.fit(train, t_train)

    def predecir(self, test):
        self.__prediccion = self.__modelo.predict(test)
        return self.__prediccion
    
    @property
    def prediccion(self):
        return self.__prediccion

def main():
    juan = modelo()
    predictores = [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]]
    etiquetas = [[1], [2], [3], [4], [5]]

    juan.entrenarModelo(predictores, etiquetas)
    juan.predecir(predictores)
    print(juan.hiperparametros)

if __name__ == "__main__":
    main()