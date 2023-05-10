#Importación de las librerías empleadas
from sklearn.model_selection import KFold
import tensorflow as tf
import pandas as pd
import numpy as np
import metricaserror as error


#Función encargada de la creación de una red neunoral feedforward.
def crearNN(n_entradas, NO= [10], FA= 'relu', LR= 0.01, LF= "mean_squared_error"):
    #Creamos la estructura de la RN.
    modelo = tf.keras.models.Sequential()
    #Definimos primero la capa de entrada de la red neuronal.
    modelo.add(tf.keras.layers.Input(shape=(n_entradas, )))
    #Definimos las capas ocultas de la red con HN neuronas. El número de capas ocultas va implícito en el número de elementos de la lista NO.
    for HN in NO:
        #Se añade cada capa con su número de neuronas y la función de activación seleccionada.
        modelo.add(tf.keras.layers.Dense(HN, activation= FA))
    #Añadimos, por último, la capa de salida. Solo posee una neurona con una función de activación lineal, puesto que es un problema de regresión.
    modelo.add(tf.keras.layers.Dense(1, activation= 'linear'))

    #Establecemos el optimizador con su tasa de aprendizaje LR.
    opt = tf.keras.optimizers.Adam(learning_rate= LR)

    #Por último, seleccionamos las características del proceso que va a seguir el entrenamiento. Por defecto, se emplea una función de error de MSE.
    modelo.compile(loss=LF, optimizer=opt, metrics=['mse'])
    return modelo

#Función encargada de realizar la validación cruzada para la RN.
def validacionCruzada(nombre, predictores, target, metricas= ["RMSE", "MeanAE"], NO= [10], FA= 'relu', LR= 0.01, LF= "mse"):
    #Establecemos el número de grupos y de iteraciones para la validación cruzada.
    CV, epochs, batch = 10, 50, 50
    n_entradas = np.shape(predictores)[1]

    #Realizamos la subdivisón de los datos en K grupos.
    kf = KFold(n_splits= CV)

    #Creamos un DataFrame con las columnas de las métricas y otro con la técnica y la iteración.
    df = pd.DataFrame(columns= metricas)
    for i, (train_index, test_index) in enumerate(kf.split(predictores, target)):
        #Dividimos los datos de entrada y salida de la RN, tanto para el entrenamiento como para el test.
        X_train, t_train = predictores.iloc[train_index, :], target.iloc[train_index]
        X_test, t_test = predictores.iloc[test_index, :], target.iloc[test_index]

        #Recargamos el modelo para resetear el valor de los pesos.
        model = crearNN(n_entradas= n_entradas, metricas= ["mae"], NO= NO, FA= FA, LR= LR, LF= LF)  

        #Entrenamos el modelo de la red neuronal.
        model.fit(X_train, t_train, epochs=epochs, batch_size=batch, verbose= 0)

        #Predecimos con el modelo entrenado para el cálculo de las métricas
        prediccion = model.predict(X_test)

        #Cargamos en el DataFrame el valor de las métricas seleccionadas.
        df.loc[i]= error.calculo(metricas, t_test, prediccion)

    #Devolvemos un DF con el DF de las métricas más dos columnas de iteración y técnica
    iteracion= [[x+1] for x in range(CV)]
    tecnica= [[nombre] for x in range(CV)]
    df2 = pd.DataFrame(np.hstack([iteracion, tecnica]), columns= ['Iteracion', 'Tecnica'])
    return pd.concat([df, df2], axis= 1, join= "inner")