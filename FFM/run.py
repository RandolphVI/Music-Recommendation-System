# -*- coding:utf-8 -*-

import subprocess
import time

start = time.time()

# cmd = 'python3 utils/count.py data/train.csv > data/fc.trva.top.txt'
# subprocess.call(cmd, shell=True)

# cmd = 'python3 utils/data_helpers.py data/fc.trva.top.txt'
# subprocess.call(cmd, shell=True)

# cmd = 'python3 utils/pre-a.py data/train.csv data/tr.gbdt.dense data/tr.gbdt.sparse'
# subprocess.call(cmd, shell=True)

# cmd = 'python3 utils/pre-a.py data/test.csv data/te.gbdt.dense data/te.gbdt.sparse'
# subprocess.call(cmd, shell=True)

# gbdt make
# cmd = 'make -C model/gbdt'
# subprocess.call(cmd, shell=True)

# cmd = 'model/gbdt/gbdt -t 35 -s 1 data/te.gbdt.dense data/te.gbdt.sparse data/tr.gbdt.dense data/tr.gbdt.sparse data/te.gbdt.out data/tr.gbdt.out'
# subprocess.call(cmd, shell=True)

# cmd = 'python3 utils/pre-b.py data/train.csv data/tr.gbdt.out data/tr.ffm'
# subprocess.call(cmd, shell=True)

# cmd = 'python3 utils/pre-b.py data/test.csv data/te.gbdt.out data/te.ffm'
# subprocess.call(cmd, shell=True)

# libffm
# `ffm-train'
# usage: ffm-train [options] training_set_file [model_file]
# options:
# -l <lambda>: set regularization parameter (default 0.00002)
# -k <factor>: set number of latent factors (default 4)
# -t <iteration>: set number of iterations (default 15)
# -r <eta>: set learning rate (default 0.2)
# -s <nr_threads>: set number of threads (default 1)
# -p <path>: set path to the validation set
# --quiet: quiet model (no output)
# --no-norm: disable instance-wise normalization
# --auto-stop: stop at the iteration that achieves the best validation loss (must be used with -p)

cmd = 'model/libffm-1.13/ffm-train -k 8 -t 30 -s 2 --auto-stop -p data/te.ffm data/tr.ffm data/model'
subprocess.call(cmd, shell=True)

# cmd = 'model/libffm-1.13/ffm-predict data/te.ffm data/model data/te.out'
# subprocess.call(cmd, shell=True)

# cmd = 'python3 utils/make_submission.py data/te.out data/submission.csv'
# subprocess.call(cmd, shell=True)

print('time used = {0:.0f} sec'.format(time.time()-start))
