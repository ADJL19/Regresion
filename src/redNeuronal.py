import tensorflow as tf
import matplotlib.pyplot as plt
import funciones
import pandas as pd

from sklearn.model_selection import KFold

#Función encargada de la creación de una red neunoral feedforward.
def crearNN(n_entradas, metricas, NO= [], FA= 'relu', LR= 0.01, LF= "mean_squared_error"):
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
    modelo.compile(loss=LF, optimizer=opt, metrics=metricas)
    return modelo

#Función encargada de realizar la validación cruzada para la RN.
def validacionCruzada(predictores, target, metricas):
    #Establecemos el número de grupos y de iteraciones para la validación cruzada.
    K, epochs = 10, 200

    #Realizamos la subdivisón de los datos en K grupos.
    kf = KFold(n_splits= K)

    #Creamos un DataFrame con las columnas de las métricas.
    df = pd.DataFrame(columns=metricas)


    for i, (train_index, test_index) in enumerate(kf.split(predictores, target)):
        #Dividimos los datos de entrada y salida de la RN, tanto para el entrenamiento como para el test.
        X_train, t_train = predictores[train_index], target[train_index]
        X_test, t_test = predictores[test_index], target[test_index]

        #Recargamos el modelo para resetear el valor de los pesos.
        model = crearNN(n_entradas= 2, metricas= metricas)  

        #Entrenamos el modelo de la red neuronal.
            # history = model.fit(X_train, t_train, validation_data=(X_test, t_test), epochs=epochs, batch_size=40, verbose=0)
        model.fit(X_train, t_train, validation_data=(X_test, t_test), epochs=epochs, batch_size=40, verbose=0)

        #Cargamos en el DataFrame el valor de las métricas seleccionadas.
        df.loc[i] = model.evaluate(X_test, t_test, batch_size=None)[1:]  

    # return results, history
    return df

def main():
    metricas = ["mean_squared_error", "mean_absolute_error"]
    path = "./data/output.csv"
    [target, predictores] = funciones.importacionDatos(path)

    r = validacionCruzada(predictores, target, metricas)
    print(r)

if __name__ == "__main__":
    main()