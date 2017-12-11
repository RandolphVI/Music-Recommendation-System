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
    if train_merged_df:
        train_merged_df.fillna({'song_length': 240,
                                'genre_ids': 'no_genre_id',
                                'language': '0',
                                'artist': 'no_artist',
                                'composer': 'no_composer',
                                'lyricist': 'no_lyricist',
                                'is_featured': '0',
                                'smaller_song': '1',
                                'song_lang_boolean': '0',
                                'artist_composer': '0',
                                'artist_composer_lyricist': '0',
                                'genre_count': 0,
                                'artist_count': 0,
                                'composer_count': 0,
                                'lyricist_count': 0,
                                'count_song_played': 1,
                                'count_artist_played': 1,
                                'gender': 'unknown',
                                'name': 'no_name',
                                'song_country': 'no_song_country',
                                'song_publisher': 'no_song_publisher',
                                'song_year': 'no_song_year'}, inplace=True)
    if test_merged_df is not None:
        test_merged_df.fillna({'song_length': 240,
                               'genre_ids': 'no_genre_id',
                               'language': '0',
                               'artist': 'no_artist',
                               'composer': 'no_composer',
                               'lyricist': 'no_lyricist',
                               'is_featured': '0',
                               'smaller_song': '1',
                               'song_lang_boolean': '0',
                               'artist_composer': '0',
                               'artist_composer_lyricist': '0',
                               'genre_count': 0,
                               'artist_count': 0,
                               'composer_count': 0,
                               'lyricist_count': 0,
                               'count_song_played': 1,
                               'count_artist_played': 1,
                               'gender': 'unknown',
                               'name': 'no_name',
                               'song_country': 'no_song_country',
                               'song_publisher': 'no_song_publisher',
                               'song_year': 'no_song_year'}, inplace=True)
