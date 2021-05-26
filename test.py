#import tensorflow as tf
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
np.random.seed(40)
import time

def main():
    start = time.time()
    m = 10
    n = 20
    train = np.random.rand(m, n) * 5
    train[train < 1] = 0
    predictions = np.zeros((m, n))

    for i in range(m):
        for j in range(n):
            j_all = np.where(train[i, :] != 0) # item that this user reviewed
            j_other = j_all[0][j_all[0] != j] # we do not include the current item in the calculation
            i_other = np.arange(m) != i # indices of ratings from other users
            review_sum = 0
            sim_sum = 0
            A_all = train[i_other, j]
            for jo in j_other: # for other items this user reviewed
                B_all = train[i_other, jo]
                j_common = np.where(((A_all != 0) & (B_all != 0))) # get user indices with both items reviewed
                if j_common[0].size: # only calculate if there exist users with both reviewed
                    A = A_all[j_common]
                    B = B_all[j_common]
                    sim = cosine_similarity(np.array([A,B]))[0,1] # only need one similarity from the returning 2x2 matrix
                    sim_sum += sim
                    review_sum += train[i, jo] * sim # ratings of other items * similarity weight
            
            predictions[i, j] = review_sum / sim_sum # divide the sum by weight sum
    
    end = time.time()
    #print(np.round(predictions))
    print("total seconds:", end-start)
            




if __name__ == "__main__":
    main()