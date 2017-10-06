import pickle
import numpy as np
import sklearn.metrics as metrics
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.linear_model import SGDClassifier
import get_training_data_script as ts

X = np.array(ts.get_features())
Y = np.array(ts.get_labels())

gnb = GaussianNB()
bnb = BernoulliNB()
mnb = MultinomialNB()
sgdc = SGDClassifier()

gnb = gnb.fit(X, Y)
model_gnb = open("final/pickle_jar/model_gnb.pkl", "wb")
pickle.dump(gnb, model_gnb)
model_gnb.close()
print("GNB training completed.")

bnb = bnb.fit(X, Y)
model_bnb = open("final/pickle_jar/model_bnb.pkl", "wb")
pickle.dump(bnb, model_bnb)
model_bnb.close()
print("BNB training completed.")

mnb = mnb.fit(X, Y)
model_mnb = open("final/pickle_jar/model_mnb.pkl", "wb")
pickle.dump(mnb, model_mnb)
model_mnb.close()
print("MNB training completed.")

sgdc = sgdc.fit(X, Y)
model_sgdc = open("final/pickle_jar/model_sgdc.pkl", "wb")
pickle.dump(sgdc, model_sgdc)
model_sgdc.close()
print("SGDC training completed.")
