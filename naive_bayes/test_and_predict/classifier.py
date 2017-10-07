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

features = test_features[50]
print(predict(features))

# Uncomment to get accuracy :-
# pl = []
# for tf in test_features:
#     pl.append(predict(tf))

#print(accuracy_score(np.array(test_labels), np.array(pl)))
