import tensorflow as tf
import matplotlib.pyplot as plt
import funciones
import pandas as pd

from sklearn.model_selection import KFold

def crearNN(n_entradas, metricas, HN= [10], FA= 'relu', LR= 0.01):
    modelo = tf.keras.models.Sequential()
    modelo.add(tf.keras.layers.Input(shape=(n_entradas, )))
    for NO in HN:
        modelo.add(tf.keras.layers.Dense(NO, activation= FA))
    modelo.add(tf.keras.layers.Dense(1, activation= 'linear'))

    opt = tf.keras.optimizers.Adam(learning_rate= LR)
    modelo.compile(loss="mean_squared_error", optimizer=opt, metrics=metricas)

    return modelo

def validacionCruzada(predictores, etiquetas, metricas):
    k = 10
    epochs = 200

    kf = KFold(n_splits= k)
    results = pd.DataFrame(columns=metricas)

    for i, (train_index, test_index) in enumerate(kf.split(predictores, etiquetas)):
        print('k_fold', i+1, 'de', k)

        # Se obtienen los paquetes de datos de entrenamiento y test en base a los índices aleatorios generados en la k-fold
        X_train, t_train = predictores[train_index], etiquetas[train_index]
        X_test, t_test = predictores[test_index], etiquetas[test_index]

        # Se carga el modelo en cada paso de la kfold para resetear el entrenamiento (pesos)
        model = crearNN(n_entradas= 2, metricas= metricas)  

        # Se realiza el entrenamiento de la red de neuronas
        history = model.fit(X_train, t_train, validation_data=(X_test, t_test), epochs=epochs, batch_size=40, verbose=0)

        # Se añade una línea en la tabla de resultados (dataframe de pandas) con los resultados de las métricas seleccionadas
        results.loc[i] = model.evaluate(X_test, t_test, batch_size=None)[1:]  

    return results, history

def main():
    metricas = ["mean_squared_error", "mean_absolute_error"]
    path="./data/output.csv"
    [etiquetas, predictores] = funciones.importacionDatos(path)
    
    # model = crearNN(2, metricas= metricas)

    [r, history] = validacionCruzada(predictores, etiquetas, metricas)

    print(r)

    # plt.plot(history.history['mean_squared_error'])
    # plt.plot(history.history['val_mean_squared_error'])
    # plt.title('Métrica de error')
    # plt.ylabel('MSE')
    # plt.xlabel('Iteración (epoch)')
    # plt.legend(['Entrenamiento', 'Test'], loc='lower right')
    # plt.show()

if __name__ == "__main__":
    main()