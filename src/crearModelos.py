from sklearn.linear_model import LinearRegression, LogisticRegression

from claseModelos import model

def RLS(config):
    return LinearRegression()



def definirModelos(config, tamPredictores= 1):
    modelos= {}
    gbl= globals().items()
    tecnicas= {nombre: funcion for nombre, funcion in gbl if callable(funcion)}

    for modelo, tf in config["modelos"].items():
        if tf: 
            modelos[modelo]= tecnicas['crear' + modelo](config[modelo])
    return modelos