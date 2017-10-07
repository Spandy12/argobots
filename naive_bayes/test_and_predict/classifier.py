import trained_models as models
from sklearn.metrics import accuracy_score
import get_testing_data_script as gt
import numpy as np
from collections import Counter

def predict(features):
    predicted_labels = []

    predicted_labels.append(models.GNB(features))
    predicted_labels.append(models.GBC(features))
    predicted_labels.append(models.RFC(features))

    most_common,num_most_common = Counter(predicted_labels).most_common(1)[0]

    return most_common

test_features = np.array(gt.get_test_features())
test_labels = np.array(gt.get_test_labels())

pl = []
for tf in test_features:
    pl.append(predict(tf))


scatter(np.array(pl), test_labels)
#print(accuracy_score(np.array(test_labels), np.array(pl)))
