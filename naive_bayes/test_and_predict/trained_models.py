import pickle

def GNB(features):
    with open("naive_bayes/pickle_jar/model_gnb.pkl", "rb") as f:
        model_gnb = pickle.load(f)
    return model_gnb.predict(features)[0]

def GBC(features):
    with open("naive_bayes/pickle_jar/model_gbc.pkl", "rb") as f:
        model_gbc = pickle.load(f)
    return model_gbc.predict(features)[0]

def RFC(features):
    with open("naive_bayes/pickle_jar/model_rfc.pkl", "rb") as f:
        model_rfc = pickle.load(f)
    return model_rfc.predict(features)[0]
