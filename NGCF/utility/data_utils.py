'''
Created on Oct 10, 2018
Tensorflow Implementation of Neural Graph Collaborative Filtering (NGCF) model in:
Wang Xiang et al. Neural Graph Collaborative Filtering. In SIGIR 2019.

@author: Xiang Wang (xiangwang@u.nus.edu)
'''
import numpy as np
import random as rd
import scipy.sparse as sp
from sklearn.model_selection import train_test_split
from time import time
import os
from pprint import pprint



def split_sparse_tensor(sp_matrix, split_ratio):
    coo_mat = sp_matrix.tocoo()
    rows = coo_mat.row
    cols = coo_mat.col
    data = coo_mat.data
    train_inds, test_inds = train_test_split(np.arange(coo_mat.getnnz()), test_size=split_ratio)
    return sp.csr_matrix((data[train_inds], (rows[train_inds], cols[train_inds])), shape= sp_matrix.shape), \
           sp.csr_matrix((data[test_inds], (rows[test_inds], cols[test_inds])), shape= sp_matrix.shape)


class Data(object):
    def __init__(self, path, batch_size, test_ratio= 0.2, val_ratio= 0.1, w_by= 10, w_ct= 3, w_clt= 3, w_clk= 1, seed= 31):
        self.path = path
        self.batch_size = batch_size
        self.test_ratio = test_ratio
        self.val_ratio = val_ratio
        #get number of users and items
        self.weight_buy = w_by
        self.weight_cart = w_ct
        self.weight_collect = w_clt
        self.weight_click = w_clk

        self.adj = {}
        self.adj['by'] = sp.load_npz(os.path.join(path, "buy_adj.npz"))
        self.adj['ct'] = sp.load_npz(os.path.join(path, "cart_adj.npz"))
        self.adj['clt'] = sp.load_npz(os.path.join(path, "collect_adj.npz"))
        self.adj['clk'] = sp.load_npz(os.path.join(path, "clk_adj.npz"))
        self.adj['sum'] = self.weight_buy *  self.adj['by'] + self.weight_cart * self.adj['ct'] + \
                                self.weight_collect * self.adj['clt'] + self.weight_click *  self.adj['clk']
        self.train_adj = {}
        self.test_adj = {}
        self.train_adj['by'], self.test_adj['by'] = split_sparse_tensor(self.adj['by'], split_ratio= test_ratio)
        self.train_adj['ct'], self.test_adj['ct'] = split_sparse_tensor(self.adj['ct'], split_ratio= test_ratio)
        self.train_adj['clt'], self.test_adj['clt'] = split_sparse_tensor(self.adj['clt'], split_ratio= test_ratio)
        self.train_adj['clk'], self.test_adj['clk'] = split_sparse_tensor(self.adj['clk'], split_ratio= test_ratio)
        self.train_adj['sum'] = self.weight_buy *  self.train_adj['by'] + self.weight_cart * self.train_adj['ct'] + \
                                self.weight_collect * self.train_adj['clt'] + self.weight_click *  self.train_adj['clk']
        self.test_adj['sum'] = self.weight_buy *  self.test_adj['by'] + self.weight_cart * self.test_adj['ct'] + \
                                self.weight_collect * self.test_adj['clt'] + self.weight_click *  self.test_adj['clk']

        self.n_users, self.n_items = self.adj['by'].shape[0], self.adj['by'].shape[1]
        self.n_train = self.train_adj['sum'].getnnz()
        self.n_test = self.test_adj['sum'].getnnz()
        self.print_statistics()
        test_user_int_count = np.squeeze(np.array(self.test_adj['sum'].sum(1)))
        self.test_users= np.arange(self.n_users)[test_user_int_count > 0]
        rd.seed(seed)
        self.rd = np.random.RandomState(seed)

    def get_adj_mat(self):
        adj_mat = {}
        norm_adj_mat = {}
        mean_adj_mat = {}
        t1 = time()
        try:
            print("reload adj matrix...")
            adj_mat['by'] = sp.load_npz(os.path.join(self.path, 'by_adj_mat.npz'))
            norm_adj_mat['by'] = sp.load_npz(os.path.join(self.path, 'by_norm_adj_mat.npz'))
            mean_adj_mat['by'] = sp.load_npz(os.path.join(self.path, 'by_mean_adj_mat.npz'))

            adj_mat['ct'] = sp.load_npz(os.path.join(self.path, 'ct_adj_mat.npz'))
            norm_adj_mat['ct'] = sp.load_npz(os.path.join(self.path, 'ct_norm_adj_mat.npz'))
            mean_adj_mat['ct'] = sp.load_npz(os.path.join(self.path, 'ct_mean_adj_mat.npz'))

            adj_mat['clt'] = sp.load_npz(os.path.join(self.path, 'clt_adj_mat.npz'))
            norm_adj_mat['clt'] = sp.load_npz(os.path.join(self.path, 'clt_norm_adj_mat.npz'))
            mean_adj_mat['clt'] = sp.load_npz(os.path.join(self.path, 'clt_mean_adj_mat.npz'))

            adj_mat['clk'] = sp.load_npz(os.path.join(self.path, 'clk_adj_mat.npz'))
            norm_adj_mat['clk'] = sp.load_npz(os.path.join(self.path, 'clk_norm_adj_mat.npz'))
            mean_adj_mat['clk'] = sp.load_npz(os.path.join(self.path, 'clk_mean_adj_mat.npz'))


        except Exception:
            print("create adj matrix...")
            adj_mat['by'], norm_adj_mat['by'], mean_adj_mat['by'] = self.create_adj_mat(self.adj['by'])
            adj_mat['ct'], norm_adj_mat['ct'], mean_adj_mat['ct'] = self.create_adj_mat(self.adj['ct'])
            adj_mat['clt'], norm_adj_mat['clt'], mean_adj_mat['clt'] = self.create_adj_mat(self.adj['clt'])
            adj_mat['clk'], norm_adj_mat['clk'], mean_adj_mat['clk'] = self.create_adj_mat(self.adj['clk'])

            sp.save_npz(os.path.join(self.path, 'by_adj_mat.npz'), adj_mat['by'])
            sp.save_npz(os.path.join(self.path, 'by_norm_adj_mat.npz'), norm_adj_mat['by'])
            sp.save_npz(os.path.join(self.path, 'by_mean_adj_mat.npz'), mean_adj_mat['by'])

            sp.save_npz(os.path.join(self.path, 'ct_adj_mat.npz'), adj_mat['ct'])
            sp.save_npz(os.path.join(self.path, 'ct_norm_adj_mat.npz'), norm_adj_mat['ct'])
            sp.save_npz(os.path.join(self.path, 'ct_mean_adj_mat.npz'), mean_adj_mat['ct'])

            sp.save_npz(os.path.join(self.path, 'clt_adj_mat.npz'), adj_mat['clt'])
            sp.save_npz(os.path.join(self.path, 'clt_norm_adj_mat.npz'), norm_adj_mat['clt'])
            sp.save_npz(os.path.join(self.path, 'clt_mean_adj_mat.npz'), mean_adj_mat['clt'])

            sp.save_npz(os.path.join(self.path, 'clk_adj_mat.npz'), adj_mat['clk'])
            sp.save_npz(os.path.join(self.path, 'clk_norm_adj_mat.npz'), norm_adj_mat['clk'])
            sp.save_npz(os.path.join(self.path, 'clk_mean_adj_mat.npz'), mean_adj_mat['clk'])

        print('already load adj matrix of buy, cart, collect, clk', adj_mat['by'].shape, time() - t1)

        return (adj_mat['by'], adj_mat['ct'], adj_mat['clt'], adj_mat['clk'], ),\
               (norm_adj_mat['by'], norm_adj_mat['ct'], norm_adj_mat['clt'], norm_adj_mat['clk'], ), \
               (mean_adj_mat['by'], mean_adj_mat['ct'], mean_adj_mat['clt'], mean_adj_mat['clk'], )

    def create_adj_mat(self, R):
        t1 = time()
        adj_mat = sp.bmat([[None, R],[R.T, None]])
        adj_mat = adj_mat.tocsr()
        print('already create adjacency matrix', adj_mat.shape, time() - t1)

        t2 = time()

        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = 1 / rowsum
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv.reshape((-1, )))

            norm_adj = d_mat_inv.dot(adj)
            # norm_adj = adj.dot(d_mat_inv)
            print('generate single-normalized adjacency matrix.')
            return norm_adj.tocoo()

        norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
        mean_adj_mat = normalized_adj_single(adj_mat)

        print('already normalize adjacency matrix', time() - t2)
        return adj_mat.tocsr(), norm_adj_mat.tocsr(), mean_adj_mat.tocsr()

    def sample_item_for_user(self, u, btype):
        if btype:
            item_array = self.train_adj[btype][u].nonzero()[1]
            candidate = self.rd.choice(item_array)
        else:
            item_set = set(self.train_adj['sum'][u].nonzero()[1])
            while True:
                candidate = self.rd.randint(0, self.n_items)  # left  cloes, right open.
                if candidate not in item_set:
                    break
        return candidate

    def sample_pair(self, user):
        pairs = (('by', 'ct'), ('by', 'clk'), ('by', None), ('ct', 'clk'), ('ct', None), ('clk', None))
        while True:
            pos, neg = pair = rd.choice(pairs)
            if self.train_adj[pos][user].getnnz() and (self.train_adj[neg][user].getnnz() if neg else True):
                break
        pos_item = self.sample_item_for_user(user, pos)
        neg_item = self.sample_item_for_user(user, neg)
        return pos_item, neg_item


    def sample(self):
        users = [self.rd.randint(0, self.n_users) for i in range(self.batch_size)]
        pairs = (('by', 'ct'), ('by', 'clk'), ('by', None), ('ct', 'clk'), ('ct', None), ('clk', None))


        pos_items, neg_items = [], []
        for i, u in enumerate(users):
            # print(i)
            while True:
                pos, neg = pair = rd.choice(pairs)
                if self.train_adj[pos][u].getnnz() and (self.train_adj[neg][u].getnnz() if neg else True):
                    break
            pos_items.append(self.sample_item_for_user(u, pos))
            neg_items.append(self.sample_item_for_user(u, neg))

        return users, pos_items, neg_items

    def print_statistics(self):
        print('n_users=%d, n_items=%d' % (self.n_users, self.n_items))
        print('n_interactions=%d' % (self.n_train + self.n_test))
        print('n_train=%d, n_test=%d, sparsity=%.5f' % (self.n_train, self.n_test, (self.n_train + self.n_test)/(self.n_users * self.n_items)))


    # TODO:
    def get_sparsity_split(self):
        try:
            split_uids, split_state = [], []
            lines = open(self.path + '/sparsity.split', 'r').readlines()

            for idx, line in enumerate(lines):
                if idx % 2 == 0:
                    split_state.append(line.strip())
                    print(line.strip())
                else:
                    split_uids.append([int(uid) for uid in line.strip().split(' ')])
            print('get sparsity split.')

        except Exception:
            split_uids, split_state = self.create_sparsity_split()
            f = open(self.path + '/sparsity.split', 'w')
            for idx in range(len(split_state)):
                f.write(split_state[idx] + '\n')
                f.write(' '.join([str(uid) for uid in split_uids[idx]]) + '\n')
            print('create sparsity split.')

        return split_uids, split_state



    def create_sparsity_split(self):
        all_users_to_test = list(self.test_set.keys())
        user_n_iid = dict()

        # generate a dictionary to store (key=n_iids, value=a list of uid).
        for uid in all_users_to_test:
            train_iids = self.train_items[uid]
            test_iids = self.test_set[uid]

            n_iids = len(train_iids) + len(test_iids)

            if n_iids not in user_n_iid.keys():
                user_n_iid[n_iids] = [uid]
            else:
                user_n_iid[n_iids].append(uid)
        split_uids = list()

        # split the whole user set into four subset.
        temp = []
        count = 1
        fold = 4
        n_count = (self.n_train + self.n_test)
        n_rates = 0

        split_state = []
        for idx, n_iids in enumerate(sorted(user_n_iid)):
            temp += user_n_iid[n_iids]
            n_rates += n_iids * len(user_n_iid[n_iids])
            n_count -= n_iids * len(user_n_iid[n_iids])

            if n_rates >= count * 0.25 * (self.n_train + self.n_test):
                split_uids.append(temp)

                state = '#inter per user<=[%d], #users=[%d], #all rates=[%d]' %(n_iids, len(temp), n_rates)
                split_state.append(state)
                print(state)

                temp = []
                n_rates = 0
                fold -= 1

            if idx == len(user_n_iid.keys()) - 1 or n_count == 0:
                split_uids.append(temp)

                state = '#inter per user<=[%d], #users=[%d], #all rates=[%d]' % (n_iids, len(temp), n_rates)
                split_state.append(state)
                print(state)



        return split_uids, split_state


if __name__ == "__main__":
    t0 = time()
    data = Data("../../Data/CIKM-toy", 1024, 0.2, 0.1, 10, 3, 3, 1, 32)
    data.get_adj_mat()
    for i in enumerate(range(20)):
        users, pos, neg = data.sample()
        pprint(users)
        pprint(pos)
        pprint(neg)
    t1= time()