# -*- coding:utf-8 -*-

import subprocess
import time

start = time.time()

# cmd = 'python3 utils/count.py dac/train.csv > dac/fc.trva.top.txt'
# subprocess.call(cmd, shell=True)
#
# cmd = 'python3 converters/data_helpers.py dac/fc.trva.top.txt'
# subprocess.call(cmd, shell=True)
#
# cmd = 'python3 converters/pre-a.py dac/train.csv dac/tr.gbdt.dense dac/tr.gbdt.sparse'
# subprocess.call(cmd, shell=True)
#
# cmd = 'python3 converters/pre-a.py dac/test.csv dac/te.gbdt.dense dac/te.gbdt.sparse'
# subprocess.call(cmd, shell=True)
#
# gbdt make
# cmd = 'make -C solvers/gbdt'
# subprocess.call(cmd, shell=True)

# cmd = 'solvers/gbdt/gbdt -t 30 -s 1 dac/te.gbdt.dense dac/te.gbdt.sparse dac/tr.gbdt.dense dac/tr.gbdt.sparse dac/te.gbdt.out dac/tr.gbdt.out'
# subprocess.call(cmd, shell=True)
#
# cmd = 'python3 converters/pre-b.py dac/train.csv dac/tr.gbdt.out dac/tr.ffm'
# subprocess.call(cmd, shell=True)
#
# cmd = 'python3 converters/pre-b.py dac/test.csv dac/te.gbdt.out dac/te.ffm'
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
# cmd = 'solvers/libffm-1.13/ffm-train -k 8 -t 30 -s 2 --auto-stop -p dac/te.ffm dac/tr.ffm dac/model'
# subprocess.call(cmd, shell=True)

# cmd = 'solvers/libffm-1.13/ffm-predict dac/te.ffm dac/model dac/te.out'
# subprocess.call(cmd, shell=True)

cmd = 'python3 utils/make_submission.py dac/te.out dac/submission.csv'
subprocess.call(cmd, shell=True)

print('time used = {0:.0f} sec'.format(time.time()-start))
