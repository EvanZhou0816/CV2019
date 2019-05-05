import numpy as np
from sklearn import metrics

def diffusion(a_m, t, k):
    '''
    
    :param a_m: n*2048
    :param t: random walk steps
    :param k: k neareat neighbor
    :return: dis-distance n*n, rank-n*k, store index in prob order
    '''

    #k in paper
    dis = metrics.pairwise.euclidean_distances(a_m)
    dis_k = closest(dis, k) #Ek in paper, has zero

    sum = np.sum(dis_k, axis=1)
    sum = sum.reshape(sum.shape[0],1)
    ori_p_k = dis_k/sum #has zero

    p_kk = ori_p_k
    while t > 0:
        t -= 1
        new_p_k = np.matmul(np.matmul(ori_p_k, p_kk), ori_p_k.T)
        p_kk = closest(new_p_k, k)

    p_kk_inf = np.where(p_kk != 0, p_kk, float('inf'))
    rank = np.argsort(p_kk_inf, axis=1)
    return dis, rank[:, :k]

def closest(a, k):
    '''
    
    :param a: np array, in distance or probability
    :param k: int, k nearest neighbor
    :return: np array, far from k nearest set to 0 
    '''
    size = np.shape(a)[0]
    order = np.argsort(a, axis=1)
    thresh_ind = order[:, k]
    thresh = a[np.array(range(size)), thresh_ind]
    thresh = thresh.reshape(size, 1)
    thresh2d = np.repeat(thresh, size, axis=1)
    a_k = np.where(a <= thresh2d, a, 0)
    I = np.eye(size)
    helper_m = np.ones((size,size)) - I
    a_k *= helper_m
    return a_k


# a = np.array([[0,0,0,0,0],[3,4,6,3,7],[0,4,9,1,6], [5,2,7,4,8]])
# d, r = diffusion(a,300,2)
# print(d)
# print(r)

ranking = diffusion(np.load('../features_matrix.npy'), 1000, 300)
np.save("ranking.npy", ranking)