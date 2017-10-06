import trained_models as models
from statistics import mode
import pickle

def predict(features):
    predicted_labels = []

    predicted_labels.append(models.GNB(features))
    predicted_labels.append(models.BNB(features))
    predicted_labels.append(models.MNB(features))
    predicted_labels.append(models.SGDC(features))

    return predicted_labels, mode(predicted_labels)

features = "3.524443,1.82035,3.069451,2.4636,5.125797,3.536822"
features = [float(v) for v in features.strip().split(",")]

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
