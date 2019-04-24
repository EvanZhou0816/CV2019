import numpy as np
from scipy.spatial import distance
from sklearn import metrics

def diffusion(a_m):
    res = metrics.pairwise.euclidean_distances(a_m)
    print(res)
    # res = np.zeros((row,row))
    # for i in range(row):
    #     for j in range(row):
    #         if i == j:
    #             res[i][j] = 0
    #         # res[i][j] = distance.euclidean(a_m[i],a_m[j])
    #         res[i][j] = np.linalg.norm(a_m[i]-a_m[j])
    #         res[j][i] = res[i][j]
    # print(res)


a = np.array([[0,0],[3,4],[0,4]])
diffusion(a)
