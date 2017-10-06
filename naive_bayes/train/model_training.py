import pickle
import numpy as np
import sklearn.metrics as metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
import get_training_data_script as ts

X = np.array(ts.get_features())
Y = np.array(ts.get_labels())

gnb = GaussianNB()
gbc = GradientBoostingClassifier()
rfc = RandomForestClassifier()

gnb = gnb.fit(X, Y)
model_gnb = open("naive_bayes/pickle_jar/model_gnb.pkl", "wb")
pickle.dump(gnb, model_gnb)
model_gnb.close()
print("GNB training completed.")

gbc = gbc.fit(X, Y)
model_gbc = open("naive_bayes/pickle_jar/model_gbc.pkl", "wb")
pickle.dump(gbc, model_gbc)
model_gbc.close()
print("GBC training completed.")

rfc = rfc.fit(X, Y)
model_rfc = open("naive_bayes/pickle_jar/model_rfc.pkl", "wb")
pickle.dump(rfc, model_rfc)
model_rfc.close()
print("RFC training completed.")
