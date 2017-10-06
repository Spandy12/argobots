import pickle

def GNB(features):
    with open("naive_bayes/pickle_jar/model_gnb.pkl", "rb") as f:
        model_gnb = pickle.load(f)
    return model_gnb.predict(features)[0]

def BNB(features):
    with open("naive_bayes/pickle_jar/model_bnb.pkl", "rb") as f:
        model_bnb = pickle.load(f)
    return model_bnb.predict(features)[0]

def MNB(features):
    with open("naive_bayes/pickle_jar/model_mnb.pkl", "rb") as f:
        model_mnb = pickle.load(f)
    return model_mnb.predict(features)[0]

def SGDC(features):
    with open("naive_bayes/pickle_jar/model_sgdc.pkl", "rb") as f:
        model_sgdc = pickle.load(f)
    return model_sgdc.predict(features)[0]

def DTC(features):
    with open("naive_bayes/pickle_jar/model_dtc.pkl", "rb") as f:
        model_dtc = pickle.load(f)
    return model_dtc.predict(features)[0]

def GBC(features):
    with open("naive_bayes/pickle_jar/model_gbc.pkl", "rb") as f:
        model_gbc = pickle.load(f)
    return model_gbc.predict(features)[0]
