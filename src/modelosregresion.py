from sklearn.linear_model import LinearRegression

class regresionLineal:
    def __init__(self):
        self.__modelo = LinearRegression()
        self.__fit_intercept = True
        self.__n_jobs = None
        self.__positive = False
        self.__predictores = []
        self.__etiquetas = []
        self.__prediccion = []

    @property
    def predictores(self):
        return self.predictores
    
    @predictores.setter
    def predictores(self, predictores):
        self.__predictores = predictores

    @property
    def etiquetas(self):
        return self.etiquetas
    
    @etiquetas.setter
    def etiquetas(self, etiquetas):
        self.__etiquetas = etiquetas 

    def modelo(self):
        return self.modelo.fit(self.predictores, self.etiquetas)

    @property
    def prediccion(self):
        self.__prediccion = self.predecir()
        return self.prediccion

    def predecir(self): 
        self.__modelo = self.modelo()
        return self.modelo.predict(self.predictores)

def main():
    juan = regresionLineal
    juan.predictores = [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]]
    juan.etiquetas = [0, 2, 4, 6, 8]
    hola = juan.etiquetas
    print(hola)

if __name__ == "__main__":
    main()