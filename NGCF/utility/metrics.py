import numpy as np
from sklearn.metrics import roc_auc_score

def recall(rank, ground_truth, N):
    return len(set(rank[:N]) & set(ground_truth)) / float(len(set(ground_truth)))


def precision_at_k(r, k):
    """Score is precision @ k
    Relevance is binary (nonzero is relevant).
    Returns:
        Precision @ k
    Raises:
        ValueError: len(r) must be >= k
    """
    assert k >= 1
    r = np.asarray(r)[:k]
    return np.mean(r, axis= -1)


def average_precision(r, cut):
    """Score is average precision (area under PR curve)
    Relevance is binary (nonzero is relevant).
    Returns:
        Average precision
    """
    r = np.asarray(r)
    out = [precision_at_k(r, k + 1) for k in range(cut) if r[k]]
    if not out:
        return 0.
    return np.sum(out)/float(min(cut, np.sum(r)))


def mean_average_precision(rs):
    """Score is mean average precision
    Relevance is binary (nonzero is relevant).
    Returns:
        Mean average precision
    """
    return np.mean([average_precision(r) for r in rs])


def dcg_at_k(r, k, method=1):
    """Score is discounted cumulative gain (dcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Returns:
        Discounted cumulative gain
    """
    r = np.asfarray(r)[:, :k]
    if r.size:
        if method == 0:
            return r[:, 0] + np.sum(r[:, 1:] / np.log2(np.arange(2, k + 1)), axis= -1)
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, k + 2)), axis= -1)
        else:
            raise ValueError('method must be 0 or 1.')
    return np.zeros((r.shape[0], ))


def ndcg_at_k(r, k, method=1):
    """Score is normalized discounted cumulative gain (ndcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Returns:
        Normalized discounted cumulative gain
    """
    dcg_max = dcg_at_k(np.sort(r, axis= -1)[:, ::-1], k, method)
    dcg_max[dcg_max == 0] = 1 # for divide zero concern
    return dcg_at_k(r, k, method) / dcg_max

def recall_at_k(r, k, all_pos_num):
    r = np.asfarray(r)[:, :k]
    assert np.all(all_pos_num > 0), "zero positive numbers. "
    return np.sum(r, axis= -1) / all_pos_num


def hit_at_k(r, k):
    r = np.array(r)[:, :k]
    return np.sum(r, axis= -1) > 0

def F1(pre, rec):
    if pre + rec > 0:
        return (2.0 * pre * rec) / (pre + rec)
    else:
        return 0.

def auc(ground_truth, prediction):
    try:
        res = roc_auc_score(y_true=ground_truth, y_score=prediction)
    except Exception:
        res = 0.
    return res