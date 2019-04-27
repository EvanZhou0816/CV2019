import numpy as np
from scipy.spatial import distance
from sklearn import metrics

def diffusion(a_m):
    dis = metrics.pairwise.euclidean_distances(a_m)
    sum = np.sum(dis, axis=1)
    sum = sum.reshape(sum.shape[0],1)
    prob = dis/sum
    prob = 1-prob
    print(prob)



    return


a = np.array([[0,0],[3,4],[0,4]])
diffusion(a)
