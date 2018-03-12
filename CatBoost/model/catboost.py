# -*- coding:utf-8 -*-

import os
import time
import xgboost
import catboost

# from catboost import CatBoostClassifier
from utils import data_helpers as dh

data_path = '../data/'
result_path = '../result/'
logs_path = '../logs/'

logger = dh.logger_fn('catlog', (logs_path + 'cat-{}.log').format(time.asctime()))


def main():
    # Preprocessing Data
    # ==================================================
    logger.info('Preprocessing data...')

    if os.path.exists(data_path + 'train.hdf') and os.path.exists(data_path + 'test.hdf'):
        pass
    else:
        dh.preprocess()

    logger.info('Done preprocessing...')

    # Creating Features
    # ==================================================
    logger.info('Creating Features...')

    file = ['Xtrain0.hdf', 'Xtrain1.hdf', 'ytrain0.hdf', 'ytrain1.hdf', 'Xtest.hdf']
    for item in file:
        if os.path.exists(data_path + item):
            pass
        else:
            dh.create_features()
            break

    logger.info('Done creating...')

    # Fitting
    # ==================================================
    logger.info('Training XGBoost & CatBoost model...')

    model_xgb = xgboost.XGBClassifier(
        learning_rate=0.03,
        max_depth=7,
        nthread=50,
        seed=1,
        n_estimators=750
    )

    model_cb = catboost.CatBoostClassifier(
        iterations=2000,
        learning_rate=0.03,
        depth=7,
        loss_function='Logloss',
        thread_count=50,
        random_seed=1
    )

    dh.train_model(model_xgb, model_cb)
    logger.info('Done training...')

    # Model Blending
    # ==================================================
    logger.info('Start models blending...')

    p = dh.blend(pct1=0.6, pct2=0.4)

    dh.make_submission(data=p)

    logger.info('Done saving model predictions...')
    logger.info('All Finished!')


if __name__ == '__main__':
    main()
