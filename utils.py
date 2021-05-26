import math
import time
import numpy as np
from numpy.core.fromnumeric import nonzero
from sklearn.metrics import ndcg_score
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
from numba import jit, cuda, njit

#np.random.seed(40)


# functionals
def prepare_table(data, dims, uid_idx, pid_idx):
    # init an empty table of ratings
    m, n = dims
    table = np.zeros((m, n))

    start = time.time()

    # fill the user-product rating table (training) (with missing values)
    for row in data.iterrows():
        user = row[1]["user-id"]
        product = row[1]["product-id"]
        rating = row[1]["rating"]
        i = uid_idx[user]
        j = pid_idx[product]
        
        table[i, j] = rating

    end = time.time()

    print("prepared %.2f seconds"%(end-start))

    return table


def generate_bitmask(user_ratings, threshold=0.8):
    ''' (1-threshold) * 100 percent of data will be selected '''
    mask = np.zeros(user_ratings.shape) != 0
    for i in range(user_ratings.shape[0]):
        candidates = np.nonzero(user_ratings[i])[0]
        if candidates.shape[0] == 1: print(i)
        #selected = (np.random.rand(candidates.shape[0]) > threshold) # WRONG
        r = int(np.floor(candidates.shape[0] * threshold))
        if r > 0:
            selected = np.random.choice(candidates, size=r, replace=False)
            #mask[i, candidates[selected]] = True
            mask[i, selected] = True
    return mask


#@jit
def my_cosine_similarity(train):
    ''' Running for loops with NumPy arrays. 
    Only calculate based on other reviewed items. 
    Compared rating vectors must not have 0s in it. 
    Uses Numba to speed up the process. '''
    m = train.shape[0]
    n = train.shape[1]
    predictions = np.zeros(train.shape)

    nonzeros = [set() for _ in range(n)]
    temp = np.nonzero(train)
    for x in range(len(temp[0])):
        i = temp[0][x]
        j = temp[1][x]
        nonzeros[j].add(i)


    for i in range(m):
        #i_other = np.arange(m) != i # indices of ratings from other users
        j_all = np.where(train[i, :] != 0) # item that this user reviewed
        for j in range(n):
            rating = train[i, j]
            if rating == 0:
                review_sum = 0
                sim_sum = 0
                j_other = j_all[0][j_all[0] != j] # we do not include the current item in the calculation
                i_in_j = nonzeros[j]
                for jo in j_other: # for other items this user reviewed
                    #B_all = train[i_other, jo]
                    i_in_jo = nonzeros[jo]
                    #j_common = np.where(((A_all != 0) & (B_all != 0))) # get user indices with both items reviewed
                    i_common = i_in_j.intersection(i_in_jo)
                    if i in i_common: i_common.remove(i)
                    i_common = list(i_common)
                    #if i_common[0].size: # only calculate if there exist users with both reviewed
                    if len(i_common) > 0:
                        A = train[i_common, j]
                        B = train[i_common, jo]
                        #sim = cosine_similarity(np.array([A,B]))[0,1] # only need one similarity from the returning 2x2 matrix
                        sim = np.dot(A, B) / (np.sqrt(np.dot(A,A)) * np.sqrt(np.dot(B,B)))
                        sim_sum += sim
                        review_sum += train[i, jo] * sim # ratings of other items * similarity weight
                
                predictions[i, j] = review_sum / (sim_sum+1e-5) # divide the sum by weight sum
            
            #else: predictions[i, j] = rating

    return predictions


def top_k_index(row, k, threshold=0, ordered=True):
    #https://stackoverflow.com/questions/6910641/how-do-i-get-indices-of-n-maximum-values-in-a-numpy-array
    idx_unsorted = np.argpartition(row, -k)[-k:]
    v_unsorted = row[idx_unsorted]
    idx_sorted = idx_unsorted[np.argsort(row[idx_unsorted])][::-1]
    #print(idx_sorted)
    theta = max([0, threshold])
    idx_selected = idx_sorted[row[idx_sorted] > theta]
    v_selected = row[idx_selected]
    return idx_selected, v_selected


def compute_scores(prediction, ground_truth, K=10, w=0, methodology=0):
    m, n = prediction.shape
    prec_list, rec_list, ndcgs = [], [], []

    if methodology: clean_mask = (ground_truth == 0) # areas without any testing data (no user data)
    else: clean_mask = np.zeros((m, n)) != 0 # whole dataset

    for i in range(m):
        pred_row = prediction[i, :]
        gt_row = ground_truth[i, :]
        
        # skip users that have no testing data
        if np.count_nonzero(pred_row) == 0 or np.count_nonzero(gt_row) == 0: 
            continue
        
        # for all predictions that are not 0, calculate stats
        notzeros = np.nonzero(pred_row)
        mu = np.mean(prediction[i,notzeros])
        sigma = np.std(prediction[i,notzeros])

        # clear all cells that have no ground truth to compare to
        prediction[i, clean_mask[i, :]] = 0
        pred = prediction[i, :]
        gt = ground_truth[i,:]
        
        tops_pred, _ = top_k_index(pred, K, mu + w*sigma)
        #tops_pred, _ = top_k_index(pred, K)
        tops_gt, v_gt = top_k_index(gt, K)

        if len(tops_pred) > 0: # precision (of all the pred, % that are gt)
            prec = len([x for x in tops_gt if x in tops_pred]) / len(tops_pred)
            prec_list.append(prec)

        if len(tops_gt) > 0: # recall (of all the gt, % that are also in pred)
            rec = len([x for x in tops_gt if x in tops_pred]) / len(tops_gt)
            rec_list.append(rec)

            if len(tops_gt) > 1: # NDCG requires >= 2 elements and ndarray in [[x1, x2, x3]]
                ndcg = ndcg_score([v_gt], [prediction[i, tops_gt]])
                ndcgs.append(ndcg)

    # compute scores
    precision = np.mean(prec_list)
    recall = np.mean(rec_list)
    F = 2 * precision * recall / (precision + recall)
    conversion_rate = np.count_nonzero(prec_list) / m
    NDCG = np.mean(ndcgs)
    
    
    return (precision, recall, F, conversion_rate, NDCG)


# arithmetic functions
def mse(a, b):
    return (np.square(a - b)).mean(axis=None)


def mae(a, b):
    return np.mean(a-b)


def sigmoid(x):
    z = math.exp(-x)
    return 1 / (1 + z)




if __name__ == "__main__":
    print("Hello World!")
    m, n = 100, 200
    ground_truth = np.random.randint(5, size=(m, n))
    sparsity = np.random.rand(m, n)
    #ground_truth[sparsity < 0.1] = 0 
    train = ground_truth.copy()
    bitmask = np.random.rand(m, n)
    train[bitmask > 0.8] = 0
    test = ground_truth - train

    start = time.time()
    #griddim = 1, 2
    #blockdim = 3, 4
    #my_cosine_similarity[griddim, blockdim](train)
    predictions = my_cosine_similarity(train)
    end = time.time()
    print(np.round(test))
    print(np.round(predictions))
    print(mae(test, predictions))
    print(mse(test, predictions))
    print("total seconds:", end-start)