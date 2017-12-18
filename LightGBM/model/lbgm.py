# -*- coding:utf-8 -*-

import gc
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from utils import data_helpers as dh

data_path = '../data/'
result_path = '../result/'
logs_path = '../logs/'

logger = dh.logger_fn('lgbmlog', (logs_path + 'lgbm-{}.log').format(time.asctime()))

# Loading Data
# ==================================================

logger.info('Loading data...')

train = pd.read_csv(data_path + 'train_lgbm.csv', dtype={'target': np.uint8,
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

test = pd.read_csv(data_path + 'test_lgbm.csv', dtype={'bd': np.uint8,
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

logger.info("Train test and validation sets...")
for col in train.columns:
    if train[col].dtype == object:
        train[col] = train[col].astype('category')
        test[col] = test[col].astype('category')

X_train = train.drop(['target'], axis=1)
y_train = train['target'].values

X_test = test.drop(['id'], axis=1)
ids = test['id'].values

logger.info('Done splitting...')

del train, test
gc.collect()

# Training Data
# ==================================================

logger.info('Training data...')

d_train_final = lgb.Dataset(data=X_train, label=y_train, max_bin=256)
watchlist_final = lgb.Dataset(data=X_train, label=y_train, max_bin=256)

NUM_ROUNDS = 3000

# Training GBDT model
logger.info('Training GBDT model...')

params_gbdt = {
    'objective': 'binary',
    'metric': {'binary_logloss', 'auc'},
    'boosting': 'gbdt',
    'learning_rate': 0.1,
    'verbose': 0,
    'num_leaves': 512,
    'bagging_fraction': 0.95,
    'bagging_freq': 1,
    'bagging_seed': 1,
    'feature_fraction': 0.9,
    'feature_fraction_seed': 1,
    'max_bin': 256,
    'max_depth': 10,
    'num_rounds': NUM_ROUNDS
}

logger.info(params_gbdt)

model_f1 = lgb.train(params_gbdt, train_set=d_train_final, valid_sets=watchlist_final, verbose_eval=5)

logger.info('Done training GBDT model and make predictions next...')

p_test_1 = model_f1.predict(X_test)

subm_gbdt = pd.DataFrame()
subm_gbdt['id'] = ids
subm_gbdt['target'] = p_test_1
subm_gbdt.to_csv(result_path + 'submission_lgbm_gbdt_' + str(params_gbdt['num_leaves']) + '_'
                 + str(NUM_ROUNDS) + '.csv', index=False, float_format='%.5f')

logger.info('Done saving GBDT model predictions...')

# Training DART model
logger.info('Training DART model...')

params_dart = {
    'objective': 'binary',
    'metric': {'binary_logloss', 'auc'},
    'boosting': 'dart',
    'learning_rate': 0.1,
    'verbose': 0,
    'num_leaves': 512,
    'bagging_fraction': 0.95,
    'bagging_freq': 1,
    'bagging_seed': 1,
    'feature_fraction': 0.9,
    'feature_fraction_seed': 1,
    'max_bin': 256,
    'max_depth': 10,
    'num_rounds': NUM_ROUNDS
}

logger.info(params_dart)

model_f2 = lgb.train(params_dart, train_set=d_train_final, valid_sets=watchlist_final, verbose_eval=5)

logger.info('Done training DART model and make predictions next...')

p_test_2 = model_f2.predict(X_test)

subm_dart = pd.DataFrame()
subm_dart['id'] = ids
subm_dart['target'] = p_test_2
subm_dart.to_csv(result_path + 'submission_lgbm_dart_' + str(params_dart['num_leaves']) + '_'
                 + str(NUM_ROUNDS) + '.csv', index=False, float_format='%.5f')

logger.info('Done saving DART model predictions...')

# Averaging two model predictions
# ==================================================

p_test_avg = np.mean([p_test_1, p_test_2], axis=0)

subm_avg = pd.DataFrame()
subm_avg['id'] = ids
subm_avg['target'] = p_test_avg
subm_avg.to_csv(result_path + 'submission_lgbm_avg_' + str(params_dart['num_leaves']) + '_'
                + str(NUM_ROUNDS) + '.csv', index=False, float_format='%.5f')

logger.info('Done saving avg predictions...')
logger.info('All Finished!')
