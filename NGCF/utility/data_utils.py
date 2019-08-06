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


        self.btype_list = ['by', 'clt', 'clk']
        self.pairs= (('by', 'clt'), ('by', 'clk'), ('by', None), ('clk', None), ('clt', 'clk'), ('clt', None))

        try:
            self.train_adj = {}
            self.test_adj = {}
            for btype in self.btype_list + ['sum', ]:
                self.train_adj[btype] = sp.load_npz(os.path.join(path, "train_adj_%s.npz" %(btype, )))
                self.test_adj[btype] = sp.load_npz(os.path.join(path, "test_adj_%s.npz" %(btype, )))

            self.n_train = self.train_adj['sum'].getnnz()
            self.n_test = self.test_adj['sum'].getnnz()

            self.n_users, self.n_items = self.train_adj['sum'].shape[0], self.train_adj['sum'].shape[1]

            self.train_items_count = {btype: np.array(self.train_adj[btype].sum(1)).reshape(-1, ).tolist() for btype in self.btype_list}
            self.train_items_per_user = {btype: self.train_adj[btype].tolil().data.tolist() for btype in self.btype_list}
            self.valid_users = np.arange(self.n_users)[np.array(self.train_adj['sum'].sum(1) > 0).reshape(-1, )]

            test_int_count = (self.test_adj['sum'] > 0).sum(1)
            self.num_test_per_user = np.array(test_int_count).reshape(-1, )
            self.test_users= np.arange(self.n_users)[self.num_test_per_user > 0]

            self.test_item_ids = np.load(os.path.join(path, "test_item_ids.npy"))
            self.test_item_rel = np.load(os.path.join(path,  "test_item_rels.npy"))
            assert len(self.test_users) == len(self.test_item_ids) == len(self.test_item_rel), "Incompatible shape."
            self.num_test_items = self.test_item_ids.shape[1]

        except FileNotFoundError as e:
            print(e, "try create negative sampling from scratch.")
            self.adj = {}
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

            for btype in self.btype_list + ['sum', ]:
                sp.save_npz(os.path.join(path, "train_adj_%s.npz" %(btype, )), self.train_adj[btype])
                sp.save_npz(os.path.join(path, "test_adj_%s.npz" %(btype, )), self.test_adj[btype])

            self.n_train = self.train_adj['sum'].getnnz()
            self.n_test = self.test_adj['sum'].getnnz()

            self.n_users, self.n_items = self.adj['sum'].shape[0], self.adj['sum'].shape[1]

            self.train_items_count = {btype: np.array(self.train_adj[btype].sum(1)).reshape(-1, ).tolist() for btype in self.btype_list}
            self.train_items_per_user = {btype: self.train_adj[btype].tolil().data.tolist() for btype in self.btype_list}
            self.valid_users = np.arange(self.n_users)[np.array(self.train_adj['sum'].sum(1) > 0).reshape(-1, )]

            test_int_count = (self.test_adj['sum'] > 0).sum(1)
            self.num_test_per_user = np.array(test_int_count).reshape(-1, )
            self.test_users= np.arange(self.n_users)[self.num_test_per_user > 0]
            """
            ************************ 构造抽样测试物品矩阵 *****************
            """
            print(self.num_test_per_user.max(), self.num_test_per_user.min())
            neg_pools = np.load(os.path.join(path, 'negative_items_pool.npy'))
            assert self.num_test_per_user.max() - self.num_test_per_user.min() < 400, "NOT enought negative items pools."
            self.num_test_items = self.num_test_per_user.min() + 400 # specific
            self.test_item_ids = np.zeros((len(self.test_users), self.num_test_items), dtype= np.int32)
            self.test_item_rel = np.zeros((len(self.test_users), self.num_test_items))
            for i, u in enumerate(self.test_users):
                u_test_behavior = self.test_adj['sum'][u]
                num_pos = u_test_behavior.getnnz()
                self.test_item_ids[i, :num_pos] = u_test_behavior.nonzero()[1]
                self.test_item_ids[i, num_pos:] = neg_pools[u][self.num_test_items - num_pos]

                self.test_item_rel[i, :num_pos] = u_test_behavior.data
                if i % 10000 == 0:
                    print( " %d / %d test users processed." %(i, len(self.test_users)))

            np.save(os.path.join(path, "test_item_ids.npy"), self.test_item_ids)
            np.save(os.path.join(path, "test_item_rels.npy"), self.test_item_rel)

            """
            *********************************
            """

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


        print('already load adj matrix of buy, cart, collect, clk in shape %s in %d seconds' %(adj_mat['by'].shape, time() - t1))

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