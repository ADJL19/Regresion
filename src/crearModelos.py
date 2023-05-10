"""Script para la creación de los modelos de regresión"""

from claseModelos import model

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from redNeuronal import crearNN

def crearRLS(modelos, config, *t):
    for fi in config["fit_intercept"]:
        for p in config["positive"]:
            modelos["RLS"+str(fi)+str(p)]= model(LinearRegression(fit_intercept= fi, positive= p))
    return modelos

def crearKNN(modelos, config, *t):
    for nn in config["vecinos"]:
        for w in config["pesos"]:
            modelos["KNN"+str(nn)+str(w)]= model(KNeighborsRegressor(n_neighbors=nn, weights=w))
    return modelos

def crearDT(modelos, config, *t):
    for c in config["criterion"]:
        for md in config["max_depth"]:
            modelos["DT"+str(c)+str(md)]= model(DecisionTreeRegressor(criterion= c, max_depth= md))
    return modelos

def crearRF(modelos, config, *t):
    for c in config["criterion"]:
        for md in config["max_depth"]:
            for a in config["arboles"]:
                modelos["RF"+str(c)+str(md)+str(a)]= model(RandomForestRegressor(criterion= c, max_depth= md, n_estimators= a, n_jobs= -1))
    return modelos

def crearSVR(modelos, config, *t):
    for c in config["C"]:
        for k in config["kernel"]:
            modelos["SVR"+str(c)+str(k)]= model(SVR(C=c, kernel=k))
    return modelos

def crearMLP(modelos, config, tamPredictores):
    for fa in config["funcion_activacion"]:
        for no in config["neuronas_ocultas"]:
            modelos["MLP"+str(fa)+str(no)]= crearNN(tamPredictores, NO= no, FA= fa)
    return modelos

def crearPN(modelos, config, *t):
    for d in config["grado"]:
        for b in config["bias"]:
            modelos["PN"+str(d)+str(b)]= model(make_pipeline(PolynomialFeatures(degree= d, include_bias= b), LinearRegression(fit_intercept= False)))
    return modelos

def crearBR(modelos, config, *t):
    for a1 in config["alpha1"]:
        for a2 in config["alpha2"]:
            for l1 in config["lambda1"]:
                for l2 in config["lambda2"]:
                    modelos["BR"+str(a1)+str(a2)+str(l1)+str(l2)]= model(BayesianRidge(alpha_1=a1, alpha_2=a2, lambda_1= l1, lambda_2= l2))
    return modelos

def definirModelos(config, tamPredictores= 1):
    modelos={}

    gbl= globals().items()
    tecnicas= {nombre: funcion for nombre, funcion in gbl if callable(funcion)}

    for modelo, tf in config["modelos"].items():
        if tf:
            if modelo == "MLP":
                modelos= tecnicas['crear' + modelo](modelos, config[modelo])
            else:
                modelos= tecnicas['crear' + modelo](modelos, config[modelo], tamPredictores)

    return modelos