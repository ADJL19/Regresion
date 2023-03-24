import tensorflow as tf
import matplotlib.pyplot as plt
import funciones
from sklearn.model_selection import train_test_split

def crearNN(n_entradas, metricas, HN= [10], FA= 'relu', LR= 0.01):
    modelo = tf.keras.models.Sequential()
    modelo.add(tf.keras.layers.Input(shape=(n_entradas, )))
    for NO in HN:
        modelo.add(tf.keras.layers.Dense(NO, activation= FA))
    modelo.add(tf.keras.layers.Dense(1, activation= 'linear'))

    opt = tf.keras.optimizers.Adam(learning_rate= LR)
    modelo.compile(loss="mean_squared_error", optimizer=opt, metrics=metricas)

    return modelo

def main():
    metricas = ["mean_squared_error", "mean_absolute_error"]
    path="./data/output.csv"
    [etiquetas, predictores] = funciones.importacionDatos(path)
    
    model = crearNN(2, metricas= metricas)

    X_train, X_test, t_train, t_test = train_test_split(predictores, etiquetas, train_size=0.7)
    history = model.fit(X_train, t_train, validation_data=(X_test, t_test), epochs=300, batch_size=40, verbose=0)

    plt.plot(history.history['mean_squared_error'])
    plt.plot(history.history['val_mean_squared_error'])
    plt.title('Métrica de error')
    plt.ylabel('MSE')
    plt.xlabel('Iteración (epoch)')
    plt.legend(['Entrenamiento', 'Test'], loc='lower right')
    plt.show()

if __name__ == "__main__":
    main()