# -*- coding:utf-8 -*-

import logging
import numpy as np
from tqdm import *


def logger_fn(name, file, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    fh = logging.FileHandler(file, mode='w')
    ch = logging.StreamHandler()
    ch.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def song_lang_boolean(x):
    # '3', '10', '24', '59' -> Chinese
    if '17.0' in str(x) or '45.0' in str(x):
        return 1
    return 0


def deal_bd_info(x):
    if int(x) >= 80 or int(x) <= 5:
        return 28
    else:
        return int(x)


def deal_source_system_tab_info(x):
    if str(x) in ['my library', 'search']:
        return 0
    elif str(x) in ['discover', 'explore', 'radio']:
        return 1
    elif str(x) in ['null', '', 'notification', 'settings']:
        return 2
    else:
        return 3


def deal_source_screen_name_info(x):
    if str(x) in ['Payment', 'My library', 'My library_Search', 'Local playlist more', 'Search']:
        return 0
    elif str(x) in ['Album more', 'Artist more', 'Concert', 'Discover Chart', 'Discover Feature', 'Discover Genre',
                    'Discover New', 'Explore', 'Radio']:
        return 1
    elif str(x) in ['People global', 'People local', 'Search Home', 'Search Trends', ' Self Profile more']:
        return 2
    else:
        return 3


def deal_source_type_info(x):
    if str(x) in ['local-library', 'local-playlist']:
        return 0
    elif str(x) in ['artist', 'album', 'my-daily-playlist', 'online-playlist', 'radio', 'song-based-playlist',
                    'top-hits-for-artist', 'topic-article-playlist', 'song']:
        return 1
    else:
        return 2


def check_missing_values(df):
    print(df.isnull().values.any())
    columns_with_nan = []
    if df.isnull().values.any():
        columns_with_nan = df.columns[df.isnull().any()].tolist()
    print(columns_with_nan)
    for col in columns_with_nan:
        print("{0} : {1}".format(col, df[col].isnull().sum()))


def deal_cat_value(x):
    if x == 'no_song_year':
        pass
    else:
        return str(x)[:-2]


def deal_nan_value(train_merged_df=None, test_merged_df=None):
    if train_merged_df is not None:
        train_merged_df.fillna({'genre_ids': 'no_genre_id',
                                'artist': 'no_artist',
                                'composer': 'no_composer',
                                'lyricist': 'no_lyricist',
                                'gender': 'unknown'}, inplace=True)
    if test_merged_df is not None:
        test_merged_df.fillna({'genre_ids': 'no_genre_id',
                               'artist': 'no_artist',
                               'composer': 'no_composer',
                               'lyricist': 'no_lyricist',
                               'gender': 'unknown'}, inplace=True)
