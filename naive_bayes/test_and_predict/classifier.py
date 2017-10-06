import trained_models as models
from statistics import mode
from sklearn.metrics import accuracy_score
#import pickle
import get_testing_data_script as gt
import numpy as np

def predict(features):
    predicted_labels = []

    predicted_labels.append(models.GNB(features))
    predicted_labels.append(models.BNB(features))
    predicted_labels.append(models.MNB(features))
    predicted_labels.append(models.SGDC(features))

    return mode(predicted_labels)

features = gt.get_test_features()[0]
print(predict(features))

# list_of_priors = []
# with open("final/pickle_jar/model_gnb.pkl", "rb") as f:
#     model = pickle.load(f)
#     list_of_priors.append(model.class_prior_)
#
# with open("final/pickle_jar/model_bnb.pkl", "rb") as f:
#     model = pickle.load(f)
#     list_of_priors.append(model.class_log_prior_)
#
# with open("final/pickle_jar/model_mnb.pkl", "rb") as f:
#     model = pickle.load(f)
#     list_of_priors.append(model.class_log_prior_)
#
# with open("final/pickle_jar/model_sgdc.pkl", "rb") as f:
#     model = pickle.load(f)
#     list_of_priors.append(model.coef_)
#
#
# print(list_of_priors)
