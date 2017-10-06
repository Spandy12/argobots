def get_features():

    features = []
    for row in open('naive_bayes/train/data.csv', 'r').read().strip().split("\n"):
        values_list = []
        for values in row.split(",")[1:]:
            values_list.append(abs(float(values)))
        features.append(values_list)

    return features

def get_labels():

    labels = []
    for row in open('naive_bayes/train/data.csv', 'r').read().strip().split("\n"):
        labels.append(row.strip().split(",")[0])

    return labels

# for f in get_features():
#     for v in f:
#         if v < 0:
#             print(v)
