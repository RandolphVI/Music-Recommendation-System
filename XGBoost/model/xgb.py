# -*- coding:utf-8 -*-
__author__ = 'Randolph'

import gc
import time
import numpy as np
import pandas as pd
import xgboost as xgb

from tqdm import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from utils import data_helpers as dh

data_path = '../data/'
result_path = '../result/'
logs_path = '../logs/'

logger = dh.logger_fn('xgblog', (logs_path + 'xgb-{0}.log').format(time.asctime()))

# Loading Data
# ==================================================

logger.info('Loading data...')

train = pd.read_csv(data_path + 'train.csv')
test = pd.read_csv(data_path + 'test.csv')

logger.info('Done loading...')

# Checking nan value
# ==================================================

logger.info('Checking nan value...')
dh.fillin_nan_value(train_merged_df=train, test_merged_df=test)
logger.info('Done checking...')

# Drop feature
# ==================================================

logger.info('Dropping data...')
drop_features = []

train = train.drop(drop_features, axis=1)
test = test.drop(drop_features, axis=1)

logger.info('Done dropping...')

# Add Features
# ==================================================

logger.info('Adding new features...')

logger.info('Done adding features...')

# Split Dataset
# ==================================================

cols = list(train.columns)
cols.remove('target')

for col in tqdm(cols):
    if train[col].dtype == 'object':
        train[col] = train[col].apply(str)
        test[col] = test[col].apply(str)

        le = LabelEncoder()
        train_vals = list(train[col].unique())
        test_vals = list(test[col].unique())
        le.fit(train_vals + test_vals)
        train[col] = le.transform(train[col])
        test[col] = le.transform(test[col])

        print(col + ': ' + str(len(train_vals)) + ', ' + str(len(test_vals)))

X = np.array(train.drop(['target'], axis=1))
y = train['target'].values

X_test = np.array(test.drop(['id'], axis=1))
ids = test['id'].values

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1, random_state=0)

d_train = xgb.DMatrix(X_train, label=y_train)
d_valid = xgb.DMatrix(X_valid, label=y_valid)
d_test = xgb.DMatrix(X_test)

eval_set = [(X_train, y_train), (X_valid, y_valid)]
watchlist = [(d_train, 'train'), (d_valid, 'valid')]

del train, test
gc.collect()

# Training Data
# ==================================================

logger.info('Training XGBoost model...')

params_xgboost = {
    'objective': 'binary:logistic',
    'eta': 0.1,
    'max_depth': 10,
    'silent': 1,
    'eval_metric': 'auc'
}

logger.info(params_xgboost)

model = xgb.train(params_xgboost, d_train, 105, watchlist, early_stopping_rounds=20, maximize=True, verbose_eval=10)

logger.info('Done training XGBoost model and make predictions next...')

p_test = model.predict(d_test)

# Preparing submission
# ==================================================

subm = pd.DataFrame()
subm['id'] = ids
subm['target'] = p_test
print(len(ids), len(p_test))
subm.to_csv(result_path + 'submission_xgboost.csv', index=False)

logger.info('Done saving XGBoost model predictions...')
logger.info('All Finished!')
