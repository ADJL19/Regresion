#Importación de las librerías empleadas
import tensorflow as tf


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
def entrenar(model, X_train, t_train, X_test):
    #Establecemos el número de grupos y de iteraciones para la validación cruzada.
    epochs, batch = 50, 50

    #Entrenamos el modelo de la red neuronal.
    model.fit(X_train, t_train, epochs=epochs, batch_size=batch, verbose= 0)

    return model.predict(X_test)