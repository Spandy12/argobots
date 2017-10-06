import pickle
import numpy as np
import sklearn.metrics as metrics
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
import get_training_data_script as ts

X = np.array(ts.get_features())
Y = np.array(ts.get_labels())

gnb = GaussianNB()
bnb = BernoulliNB()
mnb = MultinomialNB()
sgdc = SGDClassifier()
gbc = GradientBoostingClassifier()
rfc = RandomForestClassifier()

gnb = gnb.fit(X, Y)
model_gnb = open("naive_bayes/pickle_jar/model_gnb.pkl", "wb")
pickle.dump(gnb, model_gnb)
model_gnb.close()
print("GNB training completed.")

bnb = bnb.fit(X, Y)
model_bnb = open("naive_bayes/pickle_jar/model_bnb.pkl", "wb")
pickle.dump(bnb, model_bnb)
model_bnb.close()
print("BNB training completed.")

mnb = mnb.fit(X, Y)
model_mnb = open("naive_bayes/pickle_jar/model_mnb.pkl", "wb")
pickle.dump(mnb, model_mnb)
model_mnb.close()
print("MNB training completed.")

sgdc = sgdc.fit(X, Y)
model_sgdc = open("naive_bayes/pickle_jar/model_sgdc.pkl", "wb")
pickle.dump(sgdc, model_sgdc)
model_sgdc.close()
print("SGDC training completed.")

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
