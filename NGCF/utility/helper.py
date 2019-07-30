'''
Created on Aug 19, 2016
@author: Xiang Wang (xiangwang@u.nus.edu)
'''
__author__ = "xiangwang"
import os
import re
import numpy as np
import tensorflow as tf

def txt2list(file_src):
    orig_file = open(file_src, "r")
    lines = orig_file.readlines()
    return lines

def ensureDir(dir_path):
    d = os.path.dirname(dir_path)
    if not os.path.exists(d):
        os.makedirs(d)

def uni2str(unicode_str):
    return str(unicode_str.encode('ascii', 'ignore')).replace('\n', '').strip()

def hasNumbers(inputString):
    return bool(re.search(r'\d', inputString))

def delMultiChar(inputString, chars):
    for ch in chars:
        inputString = inputString.replace(ch, '')
    return inputString

def merge_two_dicts(x, y):
    z = x.copy()   # start with x's keys and values
    z.update(y)    # modifies z with y's keys and values & returns None
    return z

def early_stopping(log_value, best_value, tolerance_step, expected_order='acc', tolerance=100):
    # early stopping strategy:
    assert expected_order in ['acc', 'dec']

    if (expected_order == 'acc' and log_value >= best_value) or (expected_order == 'dec' and log_value <= best_value):
        tolerance_step = 0
        best_value = log_value
    else:
        tolerance_step += 1

    if tolerance_step >= tolerance:
        print("Early stopping is trigger at step: {} log:{}".format(tolerance, log_value))
        should_stop = True
    else:
        should_stop = False
    return best_value, tolerance_step, should_stop


def convert_to_ellpack(m):
    """

    :param m: a scipy sparse matrix
    :return:
    """
    nrows = int(m.shape[0])
    max_cols = int(m.sum(1).max())
    m_lil = m.tolil().astype(np.float32)
    rows, cols, inds, values = [], [], [], []
    for i, (col_inds, l) in enumerate(zip(m_lil.rows, m_lil.data)):
        n = len(l)
        rows.extend([i, ] * n)
        cols.extend(list(range(n)))
        inds.extend(col_inds)
        values.extend(l)

    loc_array = np.array([rows, cols], dtype= int).transpose()
    inds_array = np.array(inds)
    value_array = np.array(values)
    sp_inds = tf.SparseTensor(indices= loc_array, values= inds_array, dense_shape=(nrows, max_cols))
    sp_data = tf.SparseTensor(indices= loc_array, values= value_array, dense_shape=(nrows, max_cols))
    return sp_inds, sp_data