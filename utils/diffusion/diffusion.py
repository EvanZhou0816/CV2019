import numpy as np
import time
from sklearn import metrics


def pre_diffusion(a_m, k):
    '''

    :param a_m: n*2048
    :param k: k neareat neighbor
    :return: dis-distance n*n, rank-n*k, store index in prob order
    '''

    # k in paper
    print('in diffusion function')

    dis = metrics.pairwise.euclidean_distances(a_m)
    print('dis done')
    print(time.strftime("%H:%M:%S"))

    dis_k = closest(dis, k)  # Ek in paper, has zero
    print('dis_k done')
    print(time.strftime("%H:%M:%S"))

    sum = np.sum(dis_k, axis=1)
    print('sum done')
    print(time.strftime("%H:%M:%S"))

    sum = sum.reshape(sum.shape[0], 1)
    print('sum2 done')
    print(time.strftime("%H:%M:%S"))

    ori_p_k = dis_k / sum  # has zero
    print('ori_p_k done')
    print(time.strftime("%H:%M:%S"))

    return dis, ori_p_k

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
    helper_m = np.ones((size, size)) - I
    a_k *= helper_m
    return a_k

def randomWalk(ori_p_k, p_kk, t, k):
    p_kk = np.load(p_kk)
    ori_p_k = np.load(ori_p_k)
    while t > 0:
        print(t)
        print(time.strftime("%H:%M:%S"))
        t -= 1
        new_p_k = np.matmul(np.matmul(ori_p_k, p_kk), ori_p_k.T)
        p_kk = closest(new_p_k, k)
    return p_kk

def finalRanking(p_kk):
    p_kk_inf = np.where(p_kk != 0, p_kk, float('inf'))
    rank = np.argsort(p_kk_inf, axis=1)
    return rank

# a = np.array([[0,0,0,0,0],[3,4,6,3,7],[0,4,9,1,6], [5,2,7,4,8]])
# d, r = diffusion(a,300,2)
# print(d)
# print(r)

#dis, ori_p_k = pre_diffusion(np.load('features_matrix.npy'), 300)
#np.save("ori_p_k.npy", ori_p_k)
#np.save('euclidean_dis.npy', dis)

p_kk = randomWalk("ori_p_k.npy", "p_kk50.npy", 2, 300)
np.save("p_kk52.npy", p_kk)
