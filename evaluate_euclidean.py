import numpy as np
from sklearn import metrics
import accuracy_check

def evaluate_euclidean(feature_cnn):
    _, input_name = accuracy_check.query_import()
    rate = 0
    feature_matrix = np.load('../features_matrix.npy')
    name2index = np.load('../name2index.npy').item()
    image_size = len(feature_matrix)
    for input in input_name:
        input_index = name2index[input]
        input_feature = feature_cnn[input_index]
        distance_matrix = []
        for i in range(image_size):
            distance_matrix[i] = metrics.pairwise.euclidean_distances(input_feature, feature_matrix[i])

        distanceIndex_ranking = np.argsort(distance_matrix)
        output_index = distanceIndex_ranking[0]
        index_name = np.load("../index2name.npy").item()
        output_name = index_name[output_index]
        rate += accuracy_check.accuracy_check(input_name, output_name)
    percent = rate/image_size

    return percent

