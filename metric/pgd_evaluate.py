from sklearn.metrics import average_precision_score
from collections import defaultdict
import multiprocessing as mp
import math
import numpy as np


def _init(_test_ratings, _all_ratings, _topk_list, _predictions, _itemset):
    global test_ratings, all_ratings, topk_list, predictions, itemset
    all_ratings =defaultdict(list)
    all_ratings.update(_all_ratings)
    test_ratings = _test_ratings

    topk_list = _topk_list
    predictions = _predictions
    itemset = _itemset


def get_idcg(length):
    idcg = 0.0
    for i in range(length):
        idcg = idcg + math.log(2) / math.log(i + 2)
    return idcg


def largest_indices(ary, n):
    """Returns the n largest indices from a numpy array."""
    flat = ary.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, ary.shape)

def get_idcg(length):
    idcg = 0.0
    for i in range(length):
        idcg = idcg + math.log(2) / math.log(i + 2)
    return idcg

def get_one_performance(_uid):
    u = _uid # 处理id为_uid的user
    metrics = {}
    pos_index = list(test_ratings[u])
    pos_length = len(test_ratings[u])
    # print(u, pos_length)
    neg_index = list((itemset - set(all_ratings[u])) - set(pos_index))
    pos_index.extend(neg_index)
    pre_one = predictions[u][pos_index]

    indices = largest_indices(pre_one, topk_list[-1])
    indices = list(indices[0])
    for topk in topk_list:
        hit_value = 0
        dcg_value = 0
        for idx in range(topk):
            ranking = indices[idx]
            if ranking < pos_length:
                hit_value += 1
                dcg_value += math.log(2) / math.log(idx + 2)
        target_length = min(topk, pos_length)
        hr_cur = hit_value / target_length
        ndcg_cur = dcg_value / get_idcg(target_length)
        recall_cur = hit_value / pos_length
        precision_cur = hit_value / topk
        metrics[topk] = {'hr': hr_cur, 'ndcg': ndcg_cur, 'recall': recall_cur, 'precision': precision_cur}
    return metrics


def evaluate(_testdata, test_users, _user_items, _itemset, _topk_list, user_matrix, item_matrix, process_num):

    hr_topk_list = defaultdict(list)
    ndcg_topk_list = defaultdict(list)
    recall_topk_list = defaultdict(list)
    precision_topk_list = defaultdict(list)

    hr_out, ndcg_out = {}, {}
    recall_out, precision_out = {}, {}
    _predictions = np.matmul(user_matrix, item_matrix.T)
    with mp.Pool(processes=process_num, initializer=_init,
                 initargs=(_testdata, _user_items, _topk_list, _predictions, _itemset)) as pool:
        all_metrics = pool.map(get_one_performance, test_users)
    for i, one_metrics in enumerate(all_metrics):
        for topk in _topk_list:
            hr_topk_list[topk].append(one_metrics[topk]['hr'])
            ndcg_topk_list[topk].append(one_metrics[topk]['ndcg'])
            recall_topk_list[topk].append(one_metrics[topk]['recall'])
            precision_topk_list[topk].append(one_metrics[topk]['precision'])

    for topk in _topk_list:
        hr_out[topk] = np.mean(hr_topk_list[topk])
        ndcg_out[topk] = np.mean(ndcg_topk_list[topk])
        recall_out[topk] = np.mean(recall_topk_list[topk])
        precision_out[topk] = np.mean(precision_topk_list[topk])
    return hr_out, ndcg_out, precision_out, recall_out


def get_map(pred, label):
    '''
    计算MAP
    '''
    ap_list = []
    for i in range(label.shape[0]):
        if 1 in label[i]:
            y_true = label[i]
            y_predict = pred[i]
            precision = average_precision_score(y_true, y_predict)
            ap_list.append(precision)
    mean_ap = sum(ap_list) / len(ap_list)
    return round(mean_ap, 4)


def get_precise(pre_matrix, gt_matrix):
    '''
    计算ACC
    '''
    pre_precise = []
    for i in range(pre_matrix.shape[0]):
        if 1 in gt_matrix[i]:
            index = np.where(gt_matrix[i] == 1)[0][0]
            if pre_matrix[i][index] == max(pre_matrix[i]):
                pre_precise.append(1)
            else:
                pre_precise.append(0)
    mean_pre_precise = sum(pre_precise) / len(pre_precise)
    return round(mean_pre_precise, 4)
