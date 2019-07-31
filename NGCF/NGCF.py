'''
Created on Oct 10, 2018
Tensorflow Implementation of Neural Graph Collaborative Filtering (NGCF) model in:
Wang Xiang et al. Neural Graph Collaborative Filtering. In SIGIR 2019.

@author: Xiang Wang (xiangwang@u.nus.edu)
'''
import tensorflow as tf
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
# tf.enable_eager_execution()

from utility.helper import *
from utility.evaluate import *

class NGCF(object):
    def __init__(self,
                 adj_list,
                 user_ids,
                 item_ids,
                 global_user_attr,
                 global_item_attr,
                 data_config,
                 args,
                 pretrain_data):
        """

        :param adj_list:
        :param global_user_attr:
        :param global_item_attr:
        :param data_config:
        :param args:
        :param pretrain_data:
        """
        # argument settings
        self.model_type = 'ngcf'
        self.adj_type = args.adj_type
        self.alg_type = args.alg_type

        self.pretrain_data = pretrain_data

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']

        self.n_train_users = data_config['n_train_users']
        self.n_train_items = data_config['n_train_items']

        self.n_fold = 50

        self.user_attr_names, self.user_attr_sp_names, self.user_attr_ds_names = \
            data_config['user_attr_names'], data_config['user_sp_attr_names'], data_config['user_ds_attr_names']
        self.global_user_attr = {}
        for attr in data_config['user_sp_attr_names']:
            self.global_user_attr[attr] = tf.convert_to_tensor(global_user_attr[attr], dtype= tf.int32)
        for attr in data_config['user_ds_attr_names']:
            self.global_user_attr[attr] = tf.convert_to_tensor(global_user_attr[attr], dtype= tf.float32)

        self.item_attr_names, self.item_attr_sp_names = data_config['item_attr_names'], data_config['item_sp_attr_names']
        self.global_item_attr = {}
        for attr in data_config['item_attr_names']:
            self.global_item_attr[attr] = tf.convert_to_tensor(global_item_attr[attr], dtype= tf.int32)

        self.attr_size = {}
        for attr in data_config['user_attr_names']:
            self.attr_size[attr] = data_config['attr_size'][attr]
        for attr in data_config['item_attr_names']:
            self.attr_size[attr] = data_config['attr_size'][attr]


        self.user_attr_dim  = args.user_attr_dim
        self.user_dim = args.user_dim
        self.user_dim_sum = len(self.global_user_attr) * self.user_attr_dim + self.user_dim

        self.item_dim = args.item_dim
        self.item_attr_dim = args.item_attr_dim
        self.item_dim_sum = self.item_dim + len(self.global_item_attr) * self.item_attr_dim

        self.batch_size = args.batch_size
        self.lr = args.lr

        self.embed_size = args.embed_size

        self.weight_size = args.layer_size
        self.n_layers = len(self.weight_size)

        self.node_dropout_flag = args.node_dropout_flag

        self.model_type += '_%s_%s_l%d' % (self.adj_type, self.alg_type, self.n_layers)

        self.decay = args.regs

        self.n_relation = 4
        self.A_fold = [self._split_A_hat(adj) for adj in adj_list]
        self.item_ids = tf.convert_to_tensor(item_ids)
        self.user_ids = tf.convert_to_tensor(user_ids)

        # extract sample embedding:
        self.user_attr = {}
        for attr in self.user_attr_names:
            self.user_attr[attr] = tf.gather(self.global_user_attr[attr], self.user_ids)

        self.item_attr = {}
        for attr in self.item_attr_names:
            self.item_attr[attr] = tf.gather(self.global_item_attr[attr], self.item_ids)
        '''
        *********************************************************
        Create Placeholder for Input Data & Dropout.
        '''
        self.users =  tf.placeholder(tf.int32, shape=(None, ), name= "users")
        self.pos_items = tf.placeholder(tf.int32, shape=(None,), name= "positve_items")
        self.neg_items = tf.placeholder(tf.int32, shape=(None,), name= "negative_items")

        # self.users =  tf.placeholder(tf.int32, shape=(None,))
        # self.pos_items = tf.placeholder(tf.int32, shape=(None,))
        # self.neg_items = tf.placeholder(tf.int32, shape=(None,))
        # dropout: node dropout (adopted on the ego-networks);
        #          ... since the usage of node dropout have higher computational cost,
        #          ... please use the 'node_dropout_flag' to indicate whether use such technique.
        #          message dropout (adopted on the convolution operations).
        self.node_dropout = tf.placeholder(tf.float32, shape= (), name= "node_dropout")
        self.mess_dropout = tf.placeholder(tf.float32, shape= (), name= "mess_dropout")

        """
        *********************************************************
        Create Model Parameters (i.e., Initialize Weights).
        """
        # initialization of model parameters
        self.weights = self._init_weights()

        """
        *********************************************************
        Compute Graph-based Representations of all users & items via Message-Passing Mechanism of Graph Neural Networks.
        Different Convolutional Layers:
            1. ngcf: defined in 'Neural Graph Collaborative Filtering', SIGIR2019;
            2. gcn:  defined in 'Semi-Supervised Classification with Graph Convolutional Networks', ICLR2018;
            3. gcmc: defined in 'Graph Convolutional Matrix Completion', KDD2018;
        """
        if self.alg_type in ['ngcf']:
            self.ua_embeddings, self.ia_embeddings = self._create_ngcf_embed()

        elif self.alg_type in ['gcn']:
            self.ua_embeddings, self.ia_embeddings = self._create_gcn_embed()

        elif self.alg_type in ['gcmc']:
            self.ua_embeddings, self.ia_embeddings = self._create_gcmc_embed()

        """
        *********************************************************
        Establish the final representations for user-item pairs in batch.
        """
        self.u_g_embeddings = tf.nn.embedding_lookup(self.ua_embeddings, self.users)
        self.pos_i_g_embeddings = tf.nn.embedding_lookup(self.ia_embeddings, self.pos_items)
        self.neg_i_g_embeddings = tf.nn.embedding_lookup(self.ia_embeddings, self.neg_items)

        # self.u_g_embeddings = self.ua_embeddings[self.users]
        # self.pos_i_g_embeddings = self.ia_embeddings[self.pos_items]
        # self.neg_i_g_embeddings = self.ia_embeddings[self.neg_items]

        """
        *********************************************************
        Inference for the testing phase.
        """
        self.rating = tf.matmul(self.u_g_embeddings, self.pos_i_g_embeddings, transpose_a=False, transpose_b=True)

        """
        *********************************************************
        Generate Predictions & Optimize via BPR loss.
        """
        self.mf_loss, self.emb_loss, self.reg_loss = self.create_bpr_loss(self.u_g_embeddings,
                                                                          self.pos_i_g_embeddings,
                                                                          self.neg_i_g_embeddings)
        self.loss = self.mf_loss + self.emb_loss + self.reg_loss

        self.opt = tf.train.AdamOptimizer(learning_rate= self.lr).minimize(self.loss)

    def set_support(self, adj_list, user_ids, item_ids):
        # Generate a set of adjacency sub-matrix.
        self.A_fold = [self._split_A_hat(adj) for adj in adj_list]

        self.user_ids = tf.convert_to_tensor(user_ids)
        self.item_ids = tf.convert_to_tensor(item_ids)

        self.user_attr = {}
        for attr in self.user_attr_names:
            self.user_attr[attr]= tf.gather(self.global_user_attr[attr], self.user_ids)

        self.item_attr = {}
        for attr in self.item_attr_names:
            self.item_attr[attr]= tf.gather(self.global_item_attr[attr], self.item_ids)

    def _init_weights(self):
        all_weights = dict()

        initializer = tf.contrib.layers.xavier_initializer()

        if self.pretrain_data is None:
            all_weights['age_embedding'] = tf.Variable(initializer([self.attr_size['age'], self.user_attr_dim]), name='age_embedding')
            all_weights['career_embedding'] = tf.Variable(initializer([self.attr_size['career'], self.user_attr_dim]), name='career_embedding')
            all_weights['gender_embedding'] = tf.Variable(initializer([self.attr_size['gender'], self.user_attr_dim]), name='gender_embedding')
            all_weights['education_embedding'] = tf.Variable(initializer([self.attr_size['education'], self.user_attr_dim]), name='education_embedding')
            all_weights['income_embedding'] = tf.Variable(initializer([self.attr_size['income'], self.user_attr_dim]), name='income_embedding')
            all_weights['stage_weight'] = tf.Variable(initializer([self.attr_size['stage'], self.user_attr_dim]), name='stage_weight')
            all_weights['user_embedding'] = tf.Variable(initializer([self.n_users, self.user_dim]), name= "user_bemdding")

            all_weights['wu_embed'] = tf.Variable(initializer([self.user_dim_sum, self.embed_size]), name="user_embed_transform")
            all_weights['bu_embed'] = tf.Variable(initializer([1, self.embed_size]), name='user_embed_bias')

            all_weights['cate1_embedding'] = tf.Variable(initializer([self.attr_size['cate1'], self.item_attr_dim]), name='cate1_embedding')
            all_weights['price_embedding'] = tf.Variable(initializer([self.attr_size['price'], self.item_attr_dim]), name='item_embedding')
            all_weights['item_embedding'] = tf.Variable(initializer([self.n_items, self.item_dim]), name='item_embedding')

            all_weights['wi_embed'] = tf.Variable(initializer([self.item_dim_sum, self.embed_size]), name='item_embed_transform')
            all_weights['bi_embed'] = tf.Variable(initializer([1, self.embed_size]), name ='bi_embedding')
            print('using xavier initialization')
        else:
            all_weights['user_embedding'] = tf.Variable(initial_value=self.pretrain_data['user_embed'], trainable=True,
                                                        name='user_embedding', dtype=tf.float32)
            all_weights['item_embedding'] = tf.Variable(initial_value=self.pretrain_data['item_embed'], trainable=True,
                                                        name='item_embedding', dtype=tf.float32)
            print('using pretrained initialization')

        self.weight_size_list = [self.embed_size] + self.weight_size

        for k in range(self.n_layers):
            all_weights['W_gc_%d' %k] = [tf.Variable(
                initializer([self.weight_size_list[k], self.weight_size_list[k+1]]), name='W_gc_%d_r%d' %(k, r)) for r in range(self.n_relation)]
            all_weights['b_gc_%d' %k] = [tf.Variable(
                initializer([1, self.weight_size_list[k+1]]), name='b_gc_%d_r%d' %(k, r)) for r in range(self.n_relation)]

            all_weights['W_bi_%d' % k] = [tf.Variable(
                initializer([self.weight_size_list[k], self.weight_size_list[k + 1]]), name='W_bi_%d_r%d' %(k, r)) for r in range(self.n_relation)]
            all_weights['b_bi_%d' % k] = [tf.Variable(
                initializer([1, self.weight_size_list[k + 1]]), name='b_bi_%d_r%d' %(k, r)) for r in range(self.n_relation)]

            # TODO:
            all_weights['W_mlp_%d' % k] = tf.Variable(
                initializer([self.weight_size_list[k], self.weight_size_list[k+1]]), name='W_mlp_%d' % k)
            all_weights['b_mlp_%d' % k] = tf.Variable(
                initializer([1, self.weight_size_list[k+1]]), name='b_mlp_%d' % k)

        return all_weights

    def _split_A_hat(self, X):
        A_fold_hat = []

        fold_len = (self.n_train_users + self.n_train_items) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold -1:
                end = self.n_train_users + self.n_train_items
            else:

                end = (i_fold + 1) * fold_len

            A_fold_hat.append(convert_to_ellpack(X[start:end]))
        return A_fold_hat

    # TODO: dropout in ellpack format?
    def _split_A_hat_node_dropout(self, X):
        A_fold_hat = []

        fold_len = (self.n_train_users + self.n_train_items) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len


            if i_fold == self.n_fold -1:
                end = self.n_train_users + self.n_train_items
            else:
                end = (i_fold + 1) * fold_len

            # A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]))
            temp = self._convert_sp_mat_to_sp_tensor(X[start:end])
            n_nonzero_temp = X[start:end].count_nonzero()
            A_fold_hat.append(self._dropout_sparse(temp, 1 - self.node_dropout[0], n_nonzero_temp))

        return A_fold_hat

    def _create_user_embedding(self):
        user_embedding = {}
        for attr in self.user_attr_sp_names:
            user_embedding[attr] = tf.nn.embedding_lookup(self.weights['%s_embedding' %(attr, )], self.user_attr[attr])
        user_embedding['stage'] = tf.matmul(self.user_attr['stage'], self.weights['stage_weight'])
        user_embedding['unique'] = tf.nn.embedding_lookup(self.weights['user_embedding'], self.user_ids)

        user_embedding_concat = tf.concat([user_embedding[attr] for attr in self.user_attr_names + ('unique', )], axis= -1)
        user_rep = tf.nn.relu(tf.matmul(user_embedding_concat, self.weights['wu_embed']) + self.weights['bu_embed'])
        return user_rep

    def _create_item_embedding(self):
        item_embedding = {}
        for attr in self.item_attr_sp_names:
            item_embedding[attr] = tf.nn.embedding_lookup(self.weights['%s_embedding' %(attr, )], self.item_attr[attr])
        item_embedding['unique'] = tf.nn.embedding_lookup(self.weights['item_embedding'], self.item_ids)
        item_embedding_concat = tf.concat([item_embedding['price'], item_embedding['cate1'], item_embedding['unique']], axis= -1)
        item_rep = tf.nn.relu(tf.matmul(item_embedding_concat, self.weights['wi_embed']) + self.weights['bi_embed'])
        return item_rep

    def _create_ngcf_embed(self):



        user_embedding = self._create_user_embedding()
        item_embedding = self._create_item_embedding()

        ego_embeddings = tf.concat([user_embedding, item_embedding], axis=0)

        all_embeddings = [ego_embeddings]

        for k in range(0, self.n_layers):

            side_embeddings = []
            for r in range(self.n_relation):
                # sum messages of neighbors.
                temp_embed = []
                for f in range(self.n_fold):
                    temp_embed.append(tf.nn.embedding_lookup_sparse(ego_embeddings, *self.A_fold[r][f], combiner= 'mean'))

                embeddings = tf.concat(temp_embed, 0)
                side_embeddings.append(embeddings)

            # transformed sum messages of neighbors.
            transformed_side_embedding = [tf.matmul(side_embeddings[r], self.weights['W_gc_%d' %(k, )][r]) + self.weights['b_gc_%d' %(k, )][r]
                                          for r in range(self.n_relation)]
            sum_embeddings = tf.nn.leaky_relu(tf.add_n(transformed_side_embedding))

            # bi messages of neighbors.
            bi_embeddings = [tf.multiply(ego_embeddings, side_embeddings[r]) for r in range(self.n_relation)]
            # transformed bi messages of neighbors.
            transformed_bi_embedding = [tf.matmul(bi_embeddings[r], self.weights['W_bi_%d' %(k, )][r]) + self.weights['b_bi_%d' %(k, )][r] for r in range(self.n_relation)]
            bi_embeddings = tf.nn.leaky_relu(tf.add_n(transformed_bi_embedding))

            # non-linear activation.
            ego_embeddings = sum_embeddings + bi_embeddings

            # message dropout.
            ego_embeddings = tf.nn.dropout(ego_embeddings, 1 - self.mess_dropout)

            # normalize the distribution of embeddings.
            norm_embeddings = tf.nn.l2_normalize(ego_embeddings, axis=1)

            all_embeddings += [norm_embeddings]

        all_embeddings = tf.concat(all_embeddings, 1)
        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.n_train_users, self.n_train_items], 0)
        return u_g_embeddings, i_g_embeddings

    def _create_gcn_embed(self):
        self.A_fold = self._split_A_hat(self.adj_list)
        embeddings = tf.concat([self.weights['user_embedding'], self.weights['item_embedding']], axis=0)


        all_embeddings = [embeddings]

        for k in range(0, self.n_layers):
            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(self.A_fold[f], embeddings))

            embeddings = tf.concat(temp_embed, 0)
            embeddings = tf.nn.leaky_relu(tf.matmul(embeddings, self.weights['W_gc_%d' %k]) + self.weights['b_gc_%d' %k])
            embeddings = tf.nn.dropout(embeddings, 1 - self.mess_dropout[k])

            all_embeddings += [embeddings]

        all_embeddings = tf.concat(all_embeddings, 1)
        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.n_users, self.n_items], 0)
        return u_g_embeddings, i_g_embeddings

    def _create_gcmc_embed(self):
        # Generate a set of adjacency sub-matrix.
        # Generate a set of adjacency sub-matrix.
        if self.node_dropout_flag:
            # node dropout.
            # return a [[split slices of sparse tensor, ], ]
            A_fold_hat = self._split_A_hat_node_dropout(self.adj_list)
        else:
            A_fold_hat = self._split_A_hat(self.adj_list)

        embeddings = tf.concat([self.weights['user_embedding'], self.weights['item_embedding']], axis=0)

        all_embeddings = []

        for k in range(0, self.n_layers):
            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], embeddings))
            embeddings = tf.concat(temp_embed, 0)
            # convolutional layer.
            embeddings = tf.nn.leaky_relu(tf.matmul(embeddings, self.weights['W_gc_%d' % k]) + self.weights['b_gc_%d' % k])
            # dense layer.
            mlp_embeddings = tf.matmul(embeddings, self.weights['W_mlp_%d' %k]) + self.weights['b_mlp_%d' %k]
            mlp_embeddings = tf.nn.dropout(mlp_embeddings, 1 - self.mess_dropout[k])

            all_embeddings += [mlp_embeddings]
        all_embeddings = tf.concat(all_embeddings, 1)

        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.n_users, self.n_items], 0)
        return u_g_embeddings, i_g_embeddings


    def create_bpr_loss(self, users, pos_items, neg_items):
        pos_scores = tf.reduce_sum(tf.multiply(users, pos_items), axis=1)
        neg_scores = tf.reduce_sum(tf.multiply(users, neg_items), axis=1)

        regularizer = tf.nn.l2_loss(users) + tf.nn.l2_loss(pos_items) + tf.nn.l2_loss(neg_items)
        regularizer = regularizer/self.batch_size

        maxi = tf.log(tf.nn.sigmoid(pos_scores - neg_scores))
        mf_loss = tf.negative(tf.reduce_mean(maxi))

        emb_loss = self.decay * regularizer

        reg_loss = tf.constant(0.0, tf.float32, [1])

        return mf_loss, emb_loss, reg_loss

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        indices = np.mat([coo.row, coo.col]).transpose()
        return tf.SparseTensor(indices, coo.data, coo.shape)




    def _dropout_sparse(self, X, keep_prob, n_nonzero_elems):
        """
        Dropout for sparse tensors.
        """
        noise_shape = [n_nonzero_elems]
        random_tensor = keep_prob
        random_tensor += tf.random_uniform(noise_shape)
        dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
        pre_out = tf.sparse_retain(X, dropout_mask)

        return pre_out * tf.div(1., keep_prob)




if __name__ == '__main__':
    main()
