# -*- coding:utf-8 -*-

import gc
import time
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from utils import data_helpers as dh

data_path = '../data/'
result_path = '../result/'
logs_path = '../logs/'

logger = dh.logger_fn('xgblog', (logs_path + 'xgb-{}.log').format(time.asctime()))

# Loading Data
# ==================================================

logger.info('Loading data...')
train = pd.read_csv(data_path + 'train.csv', dtype={'target': np.uint8,
                                                    'bd': np.uint8,
                                                    'membership_days': np.uint16,
                                                    'song_length': np.uint16,
                                                    'genre_count': np.uint8,
                                                    'artist_count': np.uint8,
                                                    'composer_count': np.uint8,
                                                    'lyricist_count': np.uint8,
                                                    'count_song_played': np.uint32,
                                                    'count_artist_played': np.uint32,
                                                    'count_genre_played': np.uint32,
                                                    'listen_count': np.uint32,
                                                    'user_id': 'category',
                                                    'song_id': 'category',
                                                    'source_system_tab': 'category',
                                                    'source_screen_name': 'category',
                                                    'source_type': 'category',
                                                    'city': 'category',
                                                    'gender': 'category',
                                                    'registered_via': 'category',
                                                    'registration_year': 'category',
                                                    'registration_month': 'category',
                                                    'registration_date': 'category',
                                                    'expiration_year': 'category',
                                                    'expiration_month': 'category',
                                                    'expiration_date': 'category',
                                                    'genre_ids': 'category',
                                                    'artist': 'category',
                                                    'composer': 'category',
                                                    'lyricist': 'category',
                                                    'language': 'category',
                                                    'smaller_song': 'category',
                                                    'artist_composer': 'category',
                                                    'artist_composer_lyricist': 'category',
                                                    'song_country': 'category',
                                                    'song_publisher': 'category',
                                                    'song_year': 'category'})

test = pd.read_csv(data_path + 'test.csv', dtype={'bd': np.uint8,
                                                  'membership_days': np.uint16,
                                                  'song_length': np.uint16,
                                                  'genre_count': np.uint8,
                                                  'artist_count': np.uint8,
                                                  'composer_count': np.uint8,
                                                  'lyricist_count': np.uint8,
                                                  'count_song_played': np.uint32,
                                                  'count_artist_played': np.uint32,
                                                  'count_genre_played': np.uint32,
                                                  'listen_count': np.uint32,
                                                  'user_id': 'category',
                                                  'song_id': 'category',
                                                  'source_system_tab': 'category',
                                                  'source_screen_name': 'category',
                                                  'source_type': 'category',
                                                  'city': 'category',
                                                  'gender': 'category',
                                                  'registered_via': 'category',
                                                  'registration_year': 'category',
                                                  'registration_month': 'category',
                                                  'registration_date': 'category',
                                                  'expiration_year': 'category',
                                                  'expiration_month': 'category',
                                                  'expiration_date': 'category',
                                                  'genre_ids': 'category',
                                                  'artist': 'category',
                                                  'composer': 'category',
                                                  'lyricist': 'category',
                                                  'language': 'category',
                                                  'smaller_song': 'category',
                                                  'artist_composer': 'category',
                                                  'artist_composer_lyricist': 'category',
                                                  'song_country': 'category',
                                                  'song_publisher': 'category',
                                                  'song_year': 'category'})

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

for col in train.columns:
    if train[col].dtype == object:
        train[col] = train[col].astype('category')
        test[col] = test[col].astype('category')

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
