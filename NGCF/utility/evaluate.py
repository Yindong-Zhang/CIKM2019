'''
Created on Oct 10, 2018
Tensorflow Implementation of Neural Graph Collaborative Filtering (NGCF) model in:
Wang Xiang et al. Neural Graph Collaborative Filtering. In SIGIR 2019.

@author: Xiang Wang (xiangwang@u.nus.edu)
'''
import utility.metrics as metrics
from utility.data_utils import *
import heapq

def get_auc(item_score, user_pos_test):
    item_score = sorted(item_score.items(), key=lambda kv: kv[1])
    item_score.reverse()
    item_sort = [x[0] for x in item_score]
    posterior = [x[1] for x in item_score]

    r = []
    for i in item_sort:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = metrics.auc(ground_truth=r, prediction=posterior)
    return auc

def ranklist_by_heapq(rating, test_items, test_relevancy_vec, Ks):
    """

    :param rating: rating of all items
    :param test_items: items to test, unseen in training set
    :param test_relevancy_vec: a sparse matrix of relevancy of test_items in shape (1, num items)
    :param Ks: a list of [k, ]
    :return:
    """
    item_score = { item : rating[item] for item in test_items}

    K_max = max(Ks)
    K_max_item_inds = heapq.nlargest(K_max, item_score, key=item_score.get)

    r = [test_relevancy_vec[0, i] for i in K_max_item_inds]
    return r

def evaluate(sess, model, test_users, dataset, batchsize, K, drop_flag=False, batch_test_flag=False):
    result = {'precision': 0,
              'recall': 0,
              'ndcg': 0,
              'hit_ratio': 0}

    item_num = dataset.n_items

    u_batch_size = 10000
    i_batch_size = batchsize

    n_test_users = len(test_users)
    n_test_batch = int(np.ceil(n_test_users / u_batch_size))

    count = 0
    def evaluate_users(ratings, rel, num_pos):
        # exclude trainin
        nrows = len(ratings)
        # partial_inds = np.argpartition(ratings, -K, axis=-1)[:, -K:]
        # partial_ratings = ratings[np.arange(nrows).reshape(-1, 1), partial_inds]
        # partial_rel = dataset.test_item_rel[np.arange(nrows).reshape(-1, 1), partial_inds]


        inds = np.argsort(ratings, axis= -1)[:, -K:]
        r = rel[np.arange(nrows).reshape(-1, 1), inds]


        precision = metrics.precision_at_k(r, K)
        recall = metrics.recall_at_k(r, K, num_pos)
        ndcg = metrics.ndcg_at_k(r, K)
        hit_ratio = metrics.hit_at_k(r, K)
        return {'precision': precision.mean(),
                'recall': recall.mean(),
                'ndcg': ndcg.mean(),
                'hit_ratio': hit_ratio.mean()
                }

    for it in range(n_test_batch):
        t1 = time()
        start = it * u_batch_size
        end = min((it + 1) * u_batch_size, n_test_users)

        user_batch = test_users[start: end]

        item_batch = dataset.test_item_ids[start:end]

        if drop_flag == False:
            rate_batch = sess.run(model.rating, {model.users: user_batch,
                                                          model.test_items: item_batch})
        else:
            rate_batch = sess.run(model.rating, {model.users: user_batch,
                                                          model.test_items: item_batch,
                                                          model.node_dropout: 0,
                                                          model.mess_dropout: 0})
        t3 = time()
        num_pos = dataset.num_test_per_user[user_batch]
        rel = dataset.test_item_rel[start:end]
        tmp = evaluate_users(rate_batch, rel, num_pos)

        for indicator in 'precision', 'recall', 'ndcg', 'hit_ratio':
            result[indicator] = (result[indicator] * it + tmp[indicator]) / (it + 1)
        t2 = time()

        print("%d / %d: precision %.5f recall: %.5f ndcg %.5f hit_ratio %.5f in %d / %d seconds"
              %(it, n_test_batch, tmp['precision'], tmp['recall'], tmp['ndcg'], tmp['hit_ratio'], t2 - t1, t2 - t3))

        count += len(user_batch)

    # assert count == n_test_users
    return result
