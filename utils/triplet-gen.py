import numpy as np 
import random


"""
Helpers for generating triplets
"""


def triplet_index_gen(rank_arr):
    """
    @param:
        rank_arr: ranking matrix in 300-KNN, size=(5062,300)
    @return:
        index_knn: 2 random indices in range(300)
    """
    index_knn = []
    for row in rank_arr:
        # print(row)
        sub = random.sample(range(300), 2)
        sub.sort()
        index_knn.append(sub)
    return index_knn
    pass

if __name__ == "__main__":
    arr = [[1,2,3],[3,4,2],[4,1,3],[2,1,4],[1,2,5]]
    print(arr)
    triplet=triplet_index_gen(arr)
    print(triplet)
    # features_matrix = [[0]*2048 for i in range(5063)]
    # print(features_matrix[0])
    # change = [1]*2048
    # features_matrix[0] = change
    # print(features_matrix[0])

    pass