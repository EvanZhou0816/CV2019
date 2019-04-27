import numpy as np 
import random


"""
@ param rank_arr: ranking matrix in 300-KNN, size=(5062,300)

@ return:
"""

def triplet_gen(rank_arr):
    index_knn=[]
    for row in rank_arr:
        # print(row)
        sub = random.sample(range(300),2)
        sub.sort()
        index_knn.append(sub)
    return index_knn
    pass

if __name__ == "__main__":
    arr = [[1,2,3],[3,4,2],[4,1,3],[2,1,4],[1,2,5]]
    print(arr)
    triplet=triplet_gen(arr)
    print(triplet)
    pass