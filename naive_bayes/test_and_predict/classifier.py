import trained_models as models
from statistics import mode
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle
import get_testing_data_script as gt
import numpy as np
from collections import Counter

def predict(features):
    predicted_labels = []

    predicted_labels.append(models.GNB(features))
    predicted_labels.append(models.BNB(features))
    predicted_labels.append(models.MNB(features))
    predicted_labels.append(models.SGDC(features))
    predicted_labels.append(models.GBC(features))

    most_common,num_most_common = Counter(predicted_labels).most_common(1)[0]

    return most_common

test_features = np.array(gt.get_test_features())
test_labels = np.array(gt.get_test_labels())

pl = []
for tf in test_features:
    pl.append(predict(tf))

# print(np.array(pl))
print(accuracy_score(np.array(test_labels), np.array(pl)))
# list_of_priors = []
# with open("naive_bayes/pickle_jar/model_gnb.pkl", "rb") as f:
#     model = pickle.load(f)
#     print(model.score(np.array(test_features), np.array(test_labels)))
#
# with open("naive_bayes/pickle_jar/model_bnb.pkl", "rb") as f:
#     model = pickle.load(f)
#     print(model.score(np.array(test_features), np.array(test_labels)))
#
# with open("naive_bayes/pickle_jar/model_mnb.pkl", "rb") as f:
#     model = pickle.load(f)
#     print(model.score(np.array(test_features), np.array(test_labels)))
#
# with open("naive_bayes/pickle_jar/model_sgdc.pkl", "rb") as f:
#     model = pickle.load(f)
#     print(model.score(np.array(test_features), np.array(test_labels)))

# print(list_of_priors)
