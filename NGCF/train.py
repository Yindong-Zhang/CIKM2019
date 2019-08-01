import argparse
import os
from utility.data_utils import *
from utility.helper import *
from NGCF import NGCF
import tensorflow as tf
from utility.evaluate import  *

parser = argparse.ArgumentParser(description="Run NGCF.")
parser.add_argument('--weights_path', default='../', help='Store model path.')
parser.add_argument('--data_path', default='../Data/', help='Input data path.')
parser.add_argument('--proj_path', nargs='?', default='', help='Project path.')
parser.add_argument('--dataset', default='CIKM-toy', help='Choose a dataset from {gowalla, yelp2018, amazon-book}')
parser.add_argument('--num_sample_users', type= int, default = 1E4, help= 'Number of users to sample in single train')
parser.add_argument('--num_sample_items', type= int, default= 1E4, help= 'Number of items to sample in single traing.')
parser.add_argument('--pretrain', type=int, default=0, help='0: No pretrain, -1: Pretrain with the learned embeddings, 1:Pretrain with stored models.')
parser.add_argument('--epoch', type=int, default= 1, help='Number of epoch.')
parser.add_argument('--layer_size', nargs='+', type= int, default=[64, ], help='Output sizes of every layer')
parser.add_argument('--batch_size', type=int, default= 1024, help='Batch size.')
parser.add_argument('--regs', type= float,  default= 1e-5, help='Regularizations.')
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')
parser.add_argument('--model_type', type= str, default='ngcf', help='Specify the name of model (ngcf).')
parser.add_argument('--adj_type', type= str, default='norm', help='Specify the type of the adjacency (laplacian) matrix from {plain, norm, mean}.')
parser.add_argument('--alg_type', nargs='?', default='ngcf', help='Specify the type of the graph convolutional layer from {ngcf, gcn, gcmc}.')
parser.add_argument('--gpu_id', type=int, default= [0, 1] , help='0 for NAIS_prod, 1 for NAIS_concat')
parser.add_argument('--node_dropout_flag', type=int, default= 0, help='0: Disable node dropout, 1: Activate node dropout')
parser.add_argument('--node_dropout', type= float, default= 0.1, help='Keep probability w.r.t. node dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')
parser.add_argument('--mess_dropout', type= float, default= 0.1, help='Keep probability w.r.t. message dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')
parser.add_argument('--user_dim', type= int, default= 32, help = 'dimension of user specific vector')
parser.add_argument('--user_attr_dim', type= int, default= 16, help= 'embedding dimension for each user attr ')
parser.add_argument('--item_dim', type= int, default= 32, help= 'dimension of item specific vector')
parser.add_argument('--item_attr_dim', type= int, default= 32, help = 'embedding dimension for each item attr')
parser.add_argument('--embed_size', type=int, default=64, help='overall Embedding size for item and users.')
parser.add_argument('--K', type= int,  default= 50, help='kth first in rank performance evaluation.')
parser.add_argument('--print_every', type= int, default= 10, help= "print every several batches. ")
parser.add_argument('--save_flag', type=int, default= 1, help='0: Disable model saver, 1: Activate model saver')
parser.add_argument('--report', type=int, default=0, help='0: Disable performance report w.r.t. sparsity levels, 1: Show performance report w.r.t. sparsity levels')
args = parser.parse_args()
print(args)

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'

dataset = Data(path=os.path.join(args.data_path, args.dataset),
               batch_size=args.batch_size,
               num_sample_users= args.num_sample_users,
               num_sample_items= args.num_sample_items,
               test_ratio= 0.2,
               val_ratio= 0.1,
               adj_type= args.adj_type,
               weight= {'by': 10, 'clt': 3, 'clk': 1}
               )

data_config = dict()
data_config['n_users'] = dataset.n_users
data_config['n_items'] = dataset.n_items
data_config['n_train_users'] = dataset.num_sample_users
data_config['n_train_items'] = dataset.num_sample_items
data_config['attr_size'] = dataset.attr_size
data_config['user_attr_names'] = dataset.user_attr_names
data_config['user_sp_attr_names'] = dataset.user_sp_attr_names
data_config['user_ds_attr_names'] = dataset.user_ds_attr_names
data_config['item_attr_names'] = dataset.item_sp_attr_names
data_config['item_sp_attr_names'] = dataset.item_sp_attr_names


"""
*********************************************************
Generate the Laplacian matrix, where each entry defines the decay factor (e.g., p_ui) between two connected nodes.
"""



t0 = time()
def load_pretrained_data():
    pretrain_path = '%spretrain/%s/%s.npz' % (args.proj_path, args.dataset, 'embedding')
    try:
        pretrain_data = np.load(pretrain_path)
        print('load the pretrained embeddings.')
    except Exception:
        pretrain_data = None
    return pretrain_data

if args.pretrain == -1:
    pretrain_data = load_pretrained_data()
else:
    pretrain_data = None

adj_list = dataset.get_adj_mat()
model = NGCF(adj_list, dataset.train_users, dataset.train_items, dataset.user_attr, dataset.item_attr, data_config=data_config, args= args, pretrain_data=pretrain_data)

"""
*********************************************************
Save the model parameters.
"""
saver = tf.train.Saver()

if args.save_flag == 1:
    layer = '-'.join([str(l) for l in args.layer_size])
    weights_save_path = '%s/weights/%s/%s/%s/l%s_r%s' % (args.weights_path, args.dataset, model.model_type, layer,
                                                        args.lr, args.regs)
    ensureDir(weights_save_path)
    save_saver = tf.train.Saver(max_to_keep=1)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

"""
*********************************************************
Reload the pretrained model parameters.
"""
if args.pretrain == 1:
    layer = '-'.join([str(l) for l in args.layer_size])

    pretrain_path = '%s/weights/%s/%s/%s/l%s_r%s' % (args.weights_path, args.dataset, model.model_type, layer,
                                                    str(args.lr), args.regs)


    ckpt = tf.train.get_checkpoint_state(os.path.dirname(pretrain_path + '/checkpoint'))
    if ckpt and ckpt.model_checkpoint_path:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('load the pretrained model parameters from: ', pretrain_path)

        # *********************************************************
        # get the performance from pretrained model.
        if args.report != 1:
            res = evaluate(sess, model, dataset.test_users, dataset, args.batch_size, args.K, drop_flag=True,
                           batch_test_flag=False)
            cur_best_pre = res['recall']

            pretrain_ret = 'pretrained model recall= %.5f, precision= %.5f, hit= %.5f,' \
                           'ndcg= %.5f' % \
                           (res['recall'], res['precision'], res['hit_ratio'], res['ndcg'])
            print(pretrain_ret)
    else:
        sess.run(tf.global_variables_initializer())
        cur_best_pre = 0.
        print('without pretraining.')

else:
    sess.run(tf.global_variables_initializer())
    cur_best_pre = 0.
    print('without pretraining.')

"""
*********************************************************
Get the performance w.r.t. different sparsity levels.
"""
if args.report == 1:
    users_to_test_list, split_state = dataset.get_sparsity_split()
    users_to_test_list.append(list(dataset.test_users))
    split_state.append('all')

    report_path = '%sreport/%s/%s.result' % (args.proj_path, args.dataset, model.model_type)
    ensureDir(report_path)
    f = open(report_path, 'w')
    f.write(
        'embed_size=%d, lr=%.4f, layer_size=%s, keep_prob=%s, regs=%s, loss_type=%s, adj_type=%s\n'
        % (args.embed_size, args.lr, args.layer_size, args.keep_prob, args.regs, args.loss_type, args.adj_type))

    for i, users_to_test in enumerate(users_to_test_list):
        res = evaluate(sess, model, users_to_test, drop_flag=True)

        final_perf = "recall=[%s], precision=[%s], hit=[%s], ndcg=[%s]" % \
                     ('\t'.join(['%.5f' % r for r in res['recall']]),
                      '\t'.join(['%.5f' % r for r in res['precision']]),
                      '\t'.join(['%.5f' % r for r in res['hit_ratio']]),
                      '\t'.join(['%.5f' % r for r in res['ndcg']]))
        print(final_perf)

        f.write('\t%s\n\t%s\n' % (split_state[i], final_perf))
    f.close()
    exit()

"""
*********************************************************
Train.
"""
loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], [], []
tolerant_step = 0
should_stop = False

for epoch in range(args.epoch):
    print("Epoch %d / %d: \n" %(epoch, args.epoch))
    print("Shuffle dataset: \n")
    t1 = time()
    loss, mf_loss, emb_loss, reg_loss = 0., 0., 0., 0.
    dataset.shuffle()
    adj_list = dataset.get_adj_mat()
    train_users = dataset.train_users
    train_items = dataset.train_items
    model.set_support(adj_list, train_users, train_items)

    n_batch = dataset.n_train // args.batch_size + 1 # my choice

    print("Start training...")
    for it in range(n_batch):
        users, pos_items, neg_items = dataset.sample_batch_labels()
        _, batch_loss, batch_mf_loss, batch_emb_loss, batch_reg_loss = sess.run(
            [model.opt, model.loss, model.mf_loss, model.emb_loss, model.reg_loss],
            feed_dict={model.users: users,
                       model.pos_items: pos_items,
                       model.neg_items: neg_items,
                       model.node_dropout: args.node_dropout,
                       model.mess_dropout: args.mess_dropout,
                       })
        loss = (it * loss + batch_loss) / (it + 1)
        mf_loss = (it * mf_loss + batch_mf_loss) / (it + 1)
        emb_loss = (it * emb_loss + batch_emb_loss) / (it + 1)
        reg_loss = (it * reg_loss + batch_reg_loss) / (it + 1)

        if it % args.print_every == 0:
            print("%d / %d: loss %.4f mf loss %.4f emb loss %.4f reg loss %.4f" % (it, n_batch, loss, mf_loss, emb_loss, reg_loss))
    t2 = time()
    print("epoch %d train conclude in %d seconds." % (epoch, t2 - t1))

    print("epoch %d evaluating..." %(epoch, ))
    res = evaluate(sess, model, dataset.test_users, dataset, args.batch_size, args.K, drop_flag=True, batch_test_flag= False)

    t3 = time()
    print("epoch %d evaluate conclude in time %d seconds." %(epoch, t3 - t2))

    loss_loger.append(loss)
    rec_loger.append(res['recall'])
    pre_loger.append(res['precision'])
    ndcg_loger.append(res['ndcg'])
    hit_loger.append(res['hit_ratio'])

    perf_str = 'Epoch %d [training %d s + testing %d s]: \n' \
               'train loss= [%.5f=%.5f + %.5f + %.5f]\n' \
               'recall= %.5f\n' \
               'precision= %.5f\n' \
               'hit= %.5f\n' \
               'ndcg= %.5f\n' \
               %(epoch, t2 - t1, t3 - t2, loss, mf_loss, emb_loss, reg_loss, res['recall'], res['precision'], res['hit_ratio'], res['ndcg'])
    print(perf_str)

    cur_best_pre, tolerant_step, should_stop = early_stopping(res['recall'], cur_best_pre,
                                                              tolerant_step, expected_order='acc', tolerance=5)

    # *********************************************************
    # early stopping when cur_best_pre_0 is decreasing for ten successive steps.
    if should_stop == True:
        break

    # *********************************************************
    # save the user & item embeddings for pretraining.
    if res['recall'] == cur_best_pre and args.save_flag == 1:
        save_saver.save(sess, weights_save_path + '/weights', global_step=epoch)
        print('save the weights in path: ', weights_save_path)

recs = np.array(rec_loger)
pres = np.array(pre_loger)
ndcgs = np.array(ndcg_loger)
hit = np.array(hit_loger)

best_record = np.max(recs)
it = np.argmax(recs)
final_perf = "Best Iter=[%d]@[%.1f]\t recall= %s, precision= %s, hit= %s, ndcg= %s" % \
             (it, time() - t0,
              recs[it],
              pres[it],
              hit[it],
              ndcgs[it])
print(final_perf)

save_path = '%soutput/%s/%s.result' % (args.proj_path, args.dataset, model.model_type)
ensureDir(save_path)
with open(save_path, 'a') as f:
    f.write('embed_size=%d, lr=%.4f, layer_size=%s, node_dropout=%s, mess_dropout=%s, regs=%s, adj_type=%s\n\t%s\n'
        % (args.embed_size, args.lr, args.layer_size, args.node_dropout, args.mess_dropout, args.regs,
           args.adj_type, final_perf))