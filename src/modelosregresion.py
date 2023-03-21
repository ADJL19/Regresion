from sklearn.linear_model import LinearRegression

class regresionlineal:
    def __init__(self, fit_intercept= True, n_jobs= None, positive= False, tmodelo= LinearRegression()):
        self.__fit_intercept = fit_intercept
        self.__n_jobs = n_jobs
        self.__positive = positive
        self.__prediccion = []
        self.__tmodelo = tmodelo

    def crearModelo(self):
        self.__modelo = LinearRegression()

    def entrenarModelo(self, train, t_train):
        self.crearModelo()
        self.__modelo = self.__modelo.fit(train, t_train)

    def predecir(self, test):
        self.__prediccion = self.__modelo.predict(test)
        return self.__prediccion

def main():
    juan = regresionlineal()
    predictores = [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]]
    etiquetas = [[1], [2], [3], [4], [5]]

    juan.entrenarModelo(predictores, etiquetas)
    # print(juan.predecir([[2, 2]]))

if __name__ == "__main__":
    main()