import numpy as np
import scipy.sparse as sp
import os

path = "../Data/CIKM"
t = sp.load_npz(os.path.join(path, "train_adj_sum.npz"))
t_T = t.transpose()
s = t_T.dot(t)
print(s.shape)