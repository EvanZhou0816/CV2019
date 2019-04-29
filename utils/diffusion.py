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
    dim = dis.shape[0]

    dis_order = np.argsort(dis, axis=1)
    dis_k = np.where(dis_order<=k, dis, 0) #Ek in paper, has zero

    sum = np.sum(dis_k, axis=1)
    sum = sum.reshape(sum.shape[0],1)
    ori_p_k = dis_k/sum #has zero

    p_kk = ori_p_k
    while t > 0:
        t -= 1
        new_p_k = np.matmul(np.matmul(ori_p_k, p_kk), ori_p_k.T)
        new_order = np.argsort(new_p_k, axis=1)
        p_kk = np.where(new_order<=k, new_p_k, 0)

    p_kk_inf = np.where(p_kk != 0, p_kk, float('inf'))
    rank = np.argsort(p_kk_inf, axis=1)
    return dis, rank[:, :k]

a = np.array([[0,0,0,0,0],[3,4,6,3,7],[0,4,9,1,6], [5,2,7,4,8]])
d, r = diffusion(a,3,2)
print(d)
print(r)

# ranking = diffusion(np.load('../features_matrix.npy'), 1000, 300)
# print(ranking)