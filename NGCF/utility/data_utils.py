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
from itertools import chain



def split_sparse_tensor(sp_matrix, split_ratio):
    coo_mat = sp_matrix.tocoo()
    rows = coo_mat.row
    cols = coo_mat.col
    data = coo_mat.data
    train_inds, test_inds = train_test_split(np.arange(coo_mat.getnnz()), test_size=split_ratio)
    return sp.csr_matrix((data[train_inds], (rows[train_inds], cols[train_inds])), shape= sp_matrix.shape), \
           sp.csr_matrix((data[test_inds], (rows[test_inds], cols[test_inds])), shape= sp_matrix.shape)


class Data(object):
    def __init__(self,
                 path,
                 batch_size,
                 test_ratio= 0.2,
                 val_ratio= 0.1,
                 adj_type = 'norm',
                 weight = {'by': 10, 'clt': 5, 'clk': 1},
                 seed= 31):
        self.path = path
        self.batch_size = batch_size
        self.test_ratio = test_ratio
        self.val_ratio = val_ratio

        # random seed
        rd.seed(seed)
        self.rd = np.random.RandomState(seed)

        self.adj_type = adj_type

        #get number of users and items
        self.weight = weight

        self.adj = {}
        self.btype_list = ['by', 'clt', 'clk']
        self.pairs= (('by', 'clt'), ('by', 'clk'), ('by', None), ('clk', None), ('clt', 'clk'), ('clt', None))
        self.adj['by'] = sp.load_npz(os.path.join(path, "buy_adj.npz"))
        self.adj['clt'] = sp.load_npz(os.path.join(path, "clt_adj.npz"))
        self.adj['clk'] = sp.load_npz(os.path.join(path, "clk_adj.npz"))
        self.adj['sum'] = self.weight['by'] *  self.adj['by'] + self.weight['clt'] * self.adj['clt'] + self.weight['clk'] *  self.adj['clk']

        self.train_adj = {}
        self.test_adj = {}
        for btype in self.btype_list:
            self.train_adj[btype], self.test_adj[btype] = split_sparse_tensor(self.adj[btype], split_ratio=test_ratio)

        self.train_adj['sum'] = self.weight['by'] * self.train_adj['by'] + self.weight['clt'] * self.train_adj['clt'] + \
                                self.weight['clk'] * self.train_adj['clk']
        self.test_adj['sum'] = self.weight['by'] * self.test_adj['by'] + self.weight['clt'] * self.test_adj['clt'] + \
                               self.weight['clk'] * self.test_adj['clk']
        self.n_train = self.train_adj['sum'].getnnz()
        self.n_test = self.test_adj['sum'].getnnz()

        self.n_users, self.n_items = self.adj['by'].shape[0], self.adj['by'].shape[1]

        self.train_items_count = {btype: np.array(self.train_adj[btype].sum(1)).reshape(-1, ).tolist() for btype in self.btype_list}
        self.train_items_per_user = {btype: self.train_adj[btype].tolil().data.tolist() for btype in self.btype_list}
        self.valid_users = np.arange(self.n_users)[np.array(self.train_adj['sum'].sum(1) > 0).reshape(-1, )]

        is_int = self.test_adj['sum'] > 0
        self.num_test_per_user = np.array(is_int.sum(1)).reshape(-1, )
        self.test_users= np.arange(self.n_users)[self.num_test_per_user > 0]
        self.print_statistics()

        # attributes
        self.user_sp_attr_names = ('age', 'gender', 'career', 'income', 'education')
        self.user_ds_attr_names = ('stage', )
        self.user_attr_names = self.user_ds_attr_names + self.user_sp_attr_names

        self.user_attr = {}
        user_attr_npz = np.load(os.path.join(self.path, 'user_attrs.npz'))
        for attr in self.user_attr_names:
            if attr in self.user_sp_attr_names:
                self.user_attr[attr] = user_attr_npz[attr].reshape(-1, )
            else:
                self.user_attr[attr] = user_attr_npz[attr]

        self.item_attr_names = self.item_sp_attr_names = ('cate1', 'price')
        self.item_attr = {}
        for attr in self.item_sp_attr_names:
            item_attr_npz = np.load(os.path.join(self.path, 'item_attrs.npz'))
            self.item_attr[attr] = item_attr_npz[attr].reshape(-1, )

        self.attr_size = {}
        for attr in self.user_attr_names:
            if attr in self.user_sp_attr_names:
                self.attr_size[attr] = int(np.max(self.user_attr[attr]) + 1)
            else:
                self.attr_size[attr] = self.user_attr[attr].shape[1]

        for attr in self.item_attr_names:
            self.attr_size[attr] = int(np.max(self.item_attr[attr]) + 1)


    def get_adj_mat(self):
        adj_mat = {}
        norm_adj_mat = {}
        mean_adj_mat = {}
        t1 = time()
        try:
            for btype in self.btype_list:
                adj_mat[btype] = sp.load_npz(os.path.join(self.path, '%s_adj_mat.npz' %(btype, )))
                norm_adj_mat[btype] = sp.load_npz(os.path.join(self.path, '%s_norm_adj_mat.npz' %(btype, )))
                mean_adj_mat[btype] = sp.load_npz(os.path.join(self.path, '%s_mean_adj_mat.npz' %(btype, )))
            print("reload adj matrix...")

        except Exception:
            print("create adj matrix...")
            for btype in self.btype_list:
                adj_mat[btype], norm_adj_mat[btype], mean_adj_mat[btype] = self.create_adj_mat(self.train_adj[btype])

            for btype in self.btype_list:
                sp.save_npz(os.path.join(self.path, '%s_adj_mat.npz' %(btype, )), adj_mat[btype])
                sp.save_npz(os.path.join(self.path, '%s_norm_adj_mat.npz' %(btype, )), norm_adj_mat[btype])
                sp.save_npz(os.path.join(self.path, '%s_mean_adj_mat.npz' %(btype, )), mean_adj_mat[btype])


        print('already load adj matrix of buy, cart, collect, clk', adj_mat['by'].shape, time() - t1)

        if self.adj_type == 'plain':
            adj_list = [adj_mat[btype] for btype in self.btype_list]
            print('use the plain adjacency matrix')

        elif self.adj_type == 'norm':
            adj_list = [norm_adj_mat[btype] for btype in self.btype_list]
            print('use the normalized adjacency matrix')

        elif self.adj_type == 'gcmc':
            adj_list = [mean_adj_mat[btype] for btype in self.btype_list]
            print('use the gcmc adjacency matrix')

        else:
            adj_list = [mean_adj_mat[btype] + sp.eye(mean_adj_mat[btype].shape[0]) for btype in self.btype_list]
            print('use the mean adjacency matrix')

        return adj_list

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
            item_array = self.train_items_per_user[btype][u]
            candidate = self.rd.choice(item_array)
        else:
            item_set = set(chain(*[self.train_items_per_user[btype][u] for btype in self.btype_list]))
            while True:
                # print('u', u)
                candidate = self.rd.randint(0, self.n_items)  # left  cloes, right open.
                if candidate not in item_set:
                    break
        return candidate

    def sample_batch_labels(self):
        users = [self.rd.choice(self.valid_users) for _ in range(self.batch_size)]

        pos_items, neg_items = [], []
        for user in users:
            while True:
                pos, neg = pair = rd.choice(self.pairs)
                if self.train_items_count[pos][user] and (self.train_items_count[neg][user] if neg else True):
                    break
            pos_items.append(self.sample_item_for_user(user, pos))
            neg_items.append(self.sample_item_for_user(user, neg))

        return users, pos_items, neg_items

    def print_statistics(self):
        print('traing user %d, train_items %d.' % (self.n_users, self.n_items))
        print('n_interactions= %d for train + %d for test.' % (self.n_train, self.n_test))
        print('n_train=%d, n_test=%d, sparsity=%.5f' % (self.n_train, self.n_test, (self.n_train + self.n_test) / (self.n_users * self.n_items)))


if __name__ == "__main__":
    t0 = time()
    data = Data("../../Data/CIKM-toy",
                batch_size= 105376,
                test_ratio= 0.2,
                val_ratio= 0.1,
                seed= 32)
    data.get_adj_mat()
    user_attr = data.user_attr
    item_attr = data.item_attr
    # for i in enumerate(range(10)):
    #     users, pos, neg = data.sample_batch_labels()
    #     pprint(users)
    #     pprint(pos)
    #     pprint(neg)
    t_list = []
    for i in enumerate(range(10)):
        t0 = time()
        users, pos, neg = data.sample_batch_labels()
        t1= time()
        pprint(len(users))
        t_list.append(t1 - t0)
    print(np.mean(t_list))
    t1= time()