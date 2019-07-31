'''
Created on Oct 10, 2018
Tensorflow Implementation of Neural Graph Collaborative Filtering (NGCF) model in:
Wang Xiang et al. Neural Graph Collaborative Filtering. In SIGIR 2019.

@author: Xiang Wang (xiangwang@u.nus.edu)
'''
import utility.metrics as metrics
from utility.data_utils import *
import multiprocessing
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

def get_performance(r, Ks, num_pos):
    precision, recall, ndcg, hit_ratio = [], [], [], []

    for K in Ks:
        precision.append(metrics.precision_at_k(r, K))
        recall.append(metrics.recall_at_k(r, K, num_pos))
        ndcg.append(metrics.ndcg_at_k(r, K))
        hit_ratio.append(metrics.hit_at_k(r, K))

    return {'recall': np.array(recall), 'precision': np.array(precision),
            'ndcg': np.array(ndcg), 'hit_ratio': np.array(hit_ratio)}


def evaluate(sess, model, test_users, dataset, batchsize, K, drop_flag=False, batch_test_flag=False):
    result = {'precision': 0,
              'recall': 0,
              'ndcg': 0,
              'hit_ratio': 0}

    item_num = dataset.num_sample_items

    u_batch_size = batchsize
    i_batch_size = batchsize

    n_test_users = len(test_users)
    n_user_batchs = n_test_users // u_batch_size + 1

    count = 0

    # TODO:
    def evaluate_users(users, ratings):
        # exclude trainin
        nrows = len(users)
        train_item_rows, train_item_cols = dataset.train_adj['sum'][users].nonzero()
        ratings[train_item_rows, train_item_cols] = 0

    #     partition
        partial_inds = np.argpartition(ratings, item_num - K, axis= -1)[:, -K:]
        partial_ratings = ratings[np.arange(nrows).reshape(-1, 1), partial_inds]
        partial_rel = dataset.test_adj_sum[users.reshape(-1, 1), partial_inds].toarray()


        inds = np.argsort(partial_ratings, axis= -1)
        r = partial_rel[np.arange(nrows).reshape(-1, 1), inds]
        num_pos = dataset.num_test_per_user[users]

        precision = metrics.precision_at_k(r, K)
        recall = metrics.recall_at_k(r, K, num_pos)
        ndcg = metrics.ndcg_at_k(r, K)
        hit_ratio = metrics.hit_at_k(r, K)
        return {'precision': precision.mean(),
                'recall': recall.mean(),
                'ndcg': ndcg.mean(),
                'hit_ratio': hit_ratio.mean()
                }

    def test_one_user(x):
        # user u's ratings for user u
        rating, u = x
        # user u's items in the training set
        training_items = dataset.train_adj['sum'][u].nonzero()[1]
        # user u's items in the test set
        all_items = set(range(item_num))
        test_items = list(all_items - set(training_items))

        user_relevancy_vec = dataset.test_adj_sum[u]

        r = ranklist_by_heapq(rating, test_items, user_relevancy_vec, K)

        return get_performance(r, K, user_relevancy_vec.getnnz())

    for it in range(n_user_batchs):
        start = it * u_batch_size
        end = (it + 1) * u_batch_size

        user_batch = test_users[start: end]

        if batch_test_flag:

            n_item_batchs = item_num // i_batch_size + 1
            rate_batch = np.zeros(shape=(len(user_batch), item_num))

            i_count = 0
            for i_batch_id in range(n_item_batchs):
                i_start = i_batch_id * i_batch_size
                i_end = min((i_batch_id + 1) * i_batch_size, item_num)

                item_batch = range(i_start, i_end)

                if drop_flag == False:
                    i_rate_batch = sess.run(model.rating, {model.users: user_batch,
                                                                model.pos_items: item_batch})
                else:
                    i_rate_batch = sess.run(model.rating, {model.users: user_batch,
                                                                model.pos_items: item_batch,
                                                                model.node_dropout: 0.,
                                                                model.mess_dropout: 0.})
                rate_batch[:, i_start: i_end] = i_rate_batch
                i_count += i_rate_batch.shape[1]

            assert i_count == item_num

        else:
            item_batch = range(item_num)

            if drop_flag == False:
                rate_batch = sess.run(model.rating, {model.users: user_batch,
                                                              model.pos_items: item_batch})
            else:
                rate_batch = sess.run(model.rating, {model.users: user_batch,
                                                              model.pos_items: item_batch,
                                                              model.node_dropout: 0,
                                                              model.mess_dropout: 0})

        # user_batch_rating_uid = zip(rate_batch, user_batch)
        # batch_result = pool.map(test_one_user, user_batch_rating_uid)

        # for res in batch_result:
        #     result['precision'] += res['precision']/n_test_users
        #     result['recall'] += res['recall']/n_test_users
        #     result['ndcg'] += res['ndcg']/n_test_users
        #     result['hit_ratio'] += res['hit_ratio']/n_test_users
        tmp = evaluate_users(user_batch, rate_batch)
        for indicator in 'precision', 'recall', 'ndcg', 'hit_ratio':
            result[indicator] = (result[indicator] * it + tmp[indicator]) / (it + 1)

        count += len(user_batch)

    assert count == n_test_users
    return result
