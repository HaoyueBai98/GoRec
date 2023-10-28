from sklearn.metrics import average_precision_score
from collections import defaultdict
import multiprocessing as mp
import math
import numpy as np


def _init(_test_ratings, _all_ratings, _topk_list, _predictions, _itemset):
    global test_ratings, all_ratings, topk_list, predictions, itemset
    test_ratings = _test_ratings
    all_ratings = _all_ratings
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

    pos_index = list(test_ratings[u])
    # test_ratings是测试集的groundtruth，即test_ratings[u]是测试集中用户u交互过的全体item

    pos_length = len(test_ratings[u])
    # 测试集中用户u交互过的全体item的数量

    neg_index = list(itemset - set(all_ratings[u]))
    # all_ratings是训练集的groundtruth，即all_ratings[u]是训练集中用户u交互过的全体item
    # itemset是全部的item，所以neg_index现在表示了训练集中全体未交互过的item的编号

    pos_index.extend(neg_index)
    # 测试集的groundtruth + 训练集中未交互(评分)的负样本

    pre_one = predictions[u][pos_index]
    # predictions是模型对用户u的预测结果，对用户u来说，是一个长为item_nums（所有item总数）的向量，第i个位置表示用户u对第i个item的倾向程度
    # predictions[u][pos_index]我们不必考虑训练集中有交互的item，推荐的时候只从未推荐的内容里进行推荐，所以取出pos_index编号的item
    # 注意此时pre_one是重新排了序的，index在前pos_length，即1，2，3...，pos_length位置是用于对测试集中groundtruth的产品倾向分数

    indices = largest_indices(pre_one, topk)
    # 找出pre_one中模型给出评分最高的top_k个item

    indices = list(indices[0])

    dcg_value = 0

    topk = 5
    # 假设要计算的是NDCG@5
    for idx in range(topk):
        # 以idx=0时

        ranking = indices[idx]
        # indices[0]是模型实际给出的评分中最高的item的编号

        if ranking < pos_length:
            # pos_length是测试中groundtruth的长度，由于在pre_one的时候重新排序了，pre_one中index<pos_length都是推荐成功了的

            dcg_value += math.log(2) / math.log(idx + 2)
            # 使用的公式中 分子为 2^rel(i)-1，我们使用的rel是0或者1，所以推荐成功的时候加上一个，没命中是0

    target_length = min(topk, pos_length)
    # idcg是理想全部命中情况下的dcg，取topk或者length算一下就好了
    ndcg_cur = dcg_value / get_idcg(target_length)


def get_one_performance(_uid):
    u = _uid # 处理id为_uid的user
    metrics = {}
    pos_index = list(test_ratings[u])
    pos_length = len(test_ratings[u])
    neg_index = list(itemset - set(all_ratings[u]))
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


def evaluate(_testdata, _user_items, _topk_list, _item_count, user_matrix, item_matrix, process_num):
    _itemset = set(range(_item_count))
    hr_topk_list = defaultdict(list)
    ndcg_topk_list = defaultdict(list)
    recall_topk_list = defaultdict(list)
    precision_topk_list = defaultdict(list)

    hr_out, ndcg_out = {}, {}
    recall_out, precision_out = {}, {}
    _predictions = np.matmul(user_matrix, item_matrix.T)
    test_users = _testdata.keys()
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
    return hr_out, recall_out, ndcg_out


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
