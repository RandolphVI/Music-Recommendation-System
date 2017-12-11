# -*- coding:utf-8 -*-

import pickle
import chardet
import numpy as np
import pandas as pd
import seaborn as sbn
import matplotlib.pyplot as plt
from utils import data_helpers as dh

from tqdm import *

data_path = '../data/'
result_path = '../result/'
logs_path = '../logs/'

columns_songs = ['target', 'song_length', 'song_id', 'genre_ids', 'artist', 'composer', 'lyricist', 'language',
                 'is_featured',
                 'smaller_song', 'song_lang_boolean', 'artist_composer', 'artist_composer_lyricist', 'genre_count',
                 'artist_count', 'composer_count', 'lyricist_count', 'count_song_played', 'count_artist_played']

train_columns = ['target', 'bd', 'membership_days', 'song_length', 'genre_count', 'artist_count', 'composer_count',
                 'lyricist_count', 'count_song_played', 'count_artist_played', 'user_id', 'song_id',
                 'source_system_tab', 'source_screen_name', 'source_type', 'city', 'gender', 'registered_via',
                 'registration_year', 'registration_month', 'registration_date', 'expiration_year',
                 'expiration_month', 'expiration_date', 'genre_ids', 'artist', 'composer', 'lyricist', 'language',
                 'is_featured', 'smaller_song', 'song_lang_boolean', 'artist_composer', 'artist_composer_lyricist',
                 'name', 'song_country', 'song_publisher', 'song_year']

test_columns = ['id', 'bd', 'membership_days', 'song_length', 'genre_count', 'artist_count', 'composer_count',
                'lyricist_count', 'count_song_played', 'count_artist_played', 'user_id', 'song_id',
                'source_system_tab', 'source_screen_name', 'source_type', 'city', 'gender', 'registered_via',
                'registration_year', 'registration_month', 'registration_date', 'expiration_year',
                'expiration_month', 'expiration_date', 'genre_ids', 'artist', 'composer', 'lyricist', 'language',
                'is_featured', 'smaller_song', 'song_lang_boolean', 'artist_composer', 'artist_composer_lyricist',
                'name', 'song_country', 'song_publisher', 'song_year']

train_columns2 = ['target', 'bd', 'membership_days', 'song_length', 'genre_count', 'artist_count', 'composer_count',
                  'lyricist_count', 'count_song_played', 'count_artist_played', 'user_id', 'song_id',
                  'source_system_tab', 'source_screen_name', 'source_type', 'city', 'gender', 'registered_via',
                  'registration_year', 'registration_month', 'registration_date', 'expiration_year',
                  'expiration_month', 'expiration_date', 'genre_ids', 'artist', 'composer', 'lyricist', 'language',
                  'smaller_song', 'song_lang_boolean', 'artist_composer', 'artist_composer_lyricist', 'song_year']

test_columns2 = ['id', 'bd', 'membership_days', 'song_length', 'genre_count', 'artist_count', 'composer_count',
                 'lyricist_count', 'count_song_played', 'count_artist_played', 'user_id', 'song_id',
                 'source_system_tab', 'source_screen_name', 'source_type', 'city', 'gender', 'registered_via',
                 'registration_year', 'registration_month', 'registration_date', 'expiration_year',
                 'expiration_month', 'expiration_date', 'genre_ids', 'artist', 'composer', 'lyricist', 'language',
                 'smaller_song', 'song_lang_boolean', 'artist_composer', 'artist_composer_lyricist', 'song_year']

ffm_columns = {'bd': 'I1', 'membership_days': 'I2', 'song_length': 'I3', 'genre_count': 'I4', 'artist_count': 'I5',
               'composer_count': 'I6', 'lyricist_count': 'I7', 'count_song_played': 'I8', 'count_artist_played': 'I9',
               'user_id': 'C1', 'song_id': 'C2', 'source_system_tab': 'C3', 'source_screen_name': 'C4',
               'source_type': 'C5', 'city': 'C6', 'gender': 'C7', 'registered_via': 'C8', 'registration_year': 'C9',
               'registration_month': 'C10', 'registration_date': 'C11', 'expiration_year': 'C12',
               'expiration_month': 'C13', 'expiration_date': 'C14', 'genre_ids': 'C15', 'artist': 'C16',
               'composer': 'C17', 'lyricist': 'C18', 'language': 'C19', 'is_featured': 'C20', 'smaller_song': 'C21',
               'song_lang_boolean': 'C22', 'artist_composer': 'C23', 'artist_composer_lyricist': 'C24',
               'name': 'C25', 'song_country': 'C26', 'song_publisher': 'C27', 'song_year': 'C28'}

ffm_columns2 = {'bd': 'I1', 'membership_days': 'I2', 'song_length': 'I3', 'genre_count': 'I4', 'artist_count': 'I5',
                'composer_count': 'I6', 'lyricist_count': 'I7', 'count_song_played': 'I8', 'count_artist_played': 'I9',
                'user_id': 'C1', 'song_id': 'C2', 'source_system_tab': 'C3', 'source_screen_name': 'C4',
                'source_type': 'C5', 'city': 'C6', 'gender': 'C7', 'registered_via': 'C8', 'registration_year': 'C9',
                'registration_month': 'C10', 'registration_date': 'C11', 'expiration_year': 'C12',
                'expiration_month': 'C13', 'expiration_date': 'C14', 'genre_ids': 'C15', 'artist': 'C16',
                'composer': 'C17', 'lyricist': 'C18', 'language': 'C19', 'smaller_song': 'C20',
                'song_lang_boolean': 'C21', 'artist_composer': 'C22', 'artist_composer_lyricist': 'C23',
                'song_year': 'C24'}


def analysis_file():
    # train = pd.read_csv(data_path + 'train.csv')
    # test = pd.read_csv(data_path + 'test.csv')
    # songs = pd.read_csv(data_path + 'songs.csv')
    # members = pd.read_csv(data_path + 'members.csv', dtype={'bd': np.int16})
    # songs_extra = pd.read_csv(data_path + 'song_extra_info.csv')
    #
    # train = train.merge(songs, on='song_id', how='left')
    # test = test.merge(songs, on='song_id', how='left')
    # fuck = test[test['language'].isnull()]
    # train = train[True - train['artist'].isnull()]

    # train = train.merge(members, on='user_id', how='left')
    # test = test.merge(members, on='user_id', how='left')
    #
    # train = train.merge(songs_extra, on='song_id', how='left')
    # test = test.merge(songs_extra, on='song_id', how='left')

    # dh.deal_nan_value(train_merged_df=None, test_merged_df=test)
    # dh.check_missing_values(train)
    # dh.check_missing_values(test)

    # train.to_csv(data_path + 'train_lgbm.csv', index=False)
    # test.to_csv(data_path + 'test_lgbm.csv', index=False)

    train_lgbm = pd.read_csv(data_path + 'train_lgbm.csv')
    test_lgbm = pd.read_csv(data_path + 'test_lgbm.csv')

    # print(train_lgbm.count_artist_played.min())
    # print(train_lgbm.count_artist_played.max())
    # print(test_lgbm.count_artist_played.min())
    # print(test_lgbm.count_artist_played.max())

    # train_lgbm = train_lgbm[train_columns2]
    # test_lgbm = test_lgbm[test_columns2]
    # train_lgbm.to_csv(data_path + 'train_lgbm_2.csv', index=False)
    # test_lgbm.to_csv(data_path + 'test_lgbm_2.csv', index=False)
    #
    train_lgbm.rename(columns=ffm_columns, inplace=True)
    test_lgbm.rename(columns=ffm_columns, inplace=True)
    train_lgbm.to_csv(data_path + 'train_ffm.csv', index=False)
    test_lgbm.to_csv(data_path + 'test_ffm.csv', index=False)


def tune_file():
    def change(x):
        if not x:
            return round(float(x))
        else:
            if str(x) == 'nan':
                return int(0)
            else:
                return round(float(x))

    def change2(x):
        return round(float(x), 4)

    # members = pd.read_csv(data_path + 'members.csv')
    # song_extra = pd.read_csv(data_path + 'song_extra_info.csv')
    songs = pd.read_csv(data_path + 'songs.csv')
    # songs_new = songs_new[['song_id', 'count_song_played', 'count_song_like', 'song_like_ratio',
    #                        'count_genre_ids_played', 'count_genre_ids_like', 'genre_ids_like_ratio',
    #                        'count_artist_played', 'count_artist_like', 'artist_like_ratio',
    #                        'count_composer_played', 'count_composer_like', 'composer_like_ratio',
    #                        'count_lyricist_played', 'count_lyricist_like', 'lyricist_like_ratio']]
    # songs_new = songs_new[['song_id', 'count_song_played', 'count_genre_ids_played',
    #                        'count_artist_played', 'count_composer_played', 'count_lyricist_played']]
    # songs = songs_new.merge(songs, on='song_id', how='left')
    # songs['count_song_played'] = songs['count_song_played'].apply(change).astype(np.uint32)
    # songs['count_genre_played'] = songs['count_genre_ids_played'].apply(change).astype(np.uint32)
    # songs['count_artist_played'] = songs['count_artist_played'].apply(change).astype(np.uint32)
    # songs['count_composer_played'] = songs['count_composer_played'].apply(change).astype(np.uint32)
    # songs['count_lyricist_played'] = songs['count_lyricist_played'].apply(change).astype(np.uint32)
    #
    # songs.to_csv(data_path + 'songs_new.csv', index=False, columns=columns_songs)

    songs = songs.drop([], axis=1)
    songs.to_csv(data_path + 'songs_new.csv', index=False, columns=columns_songs)

    # train = pd.read_csv(data_path + 'train.csv')
    # test = pd.read_csv(data_path + 'test.csv')
    # train_new = pd.read_csv(data_path + 'train_new.csv')
    # test_new = pd.read_csv(data_path + 'test_new.csv')
    #
    # train.drop(['source_system_tab', 'source_screen_name', 'source_type'], axis=1)
    # test.drop(['source_system_tab', 'source_screen_name', 'source_type'], axis=1)
    #
    # train['source_system_tab'] = train_new['source_system_tab']
    # test['source_system_tab'] = test_new['source_system_tab']
    #
    # train['source_screen_name'] = train_new['source_screen_name']
    # test['source_screen_name'] = test_new['source_screen_name']
    #
    # train['source_type'] = train_new['source_type']
    # test['source_type'] = test_new['source_type']
    #
    # train.to_csv(data_path + 'train_new2.csv', index=False)
    # test.to_csv(data_path + 'test_new2.csv', index=False)


def load_pickle_file():
    with open(data_path + 'song_like_dic.pkl', 'rb') as fin1, open(data_path + 'artist_dic_new.pkl', 'rb') as fin2:
        songs = pickle.load(fin1)
        artist = pickle.load(fin2)
        try:
            print(songs['a0S959XXxdHq02RRyJIYYysamjxwJNmoCnkadrWVrjA='])
        except:
            pass
        try:
            print(artist['Phil Collins'])
        except:
            pass


def ensemble(percent_ffm=0.2, percent_xgboost=0.3):
    lgbm = pd.read_csv(result_path + 'submission_lgbm_avg.csv')
    xgboost = pd.read_csv(result_path + 'submission_xgboost.csv')
    ffm = pd.read_csv(result_path + 'submission_ffm.csv')
    avg = pd.read_csv(result_path + 'submission_final.csv')

    p_test_avg = percent_ffm * ffm['target'] + percent_xgboost * xgboost['target'] \
                 + (1 - percent_ffm - percent_xgboost) * lgbm['target']

    subm_avg = pd.DataFrame()
    subm_avg['id'] = lgbm['id']
    subm_avg['target'] = p_test_avg
    subm_avg.to_csv(result_path + 'submission_final_ffm_xgboost_lgbm.csv', index=False, float_format='%.5f')


def detect(file):
    with open(file, "rb") as f:
        data = f.read()
    print(chardet.detect(data))


if __name__ == '__main__':
    # tune_file()
    analysis_file()
    # load_pickle_file()
    # ensemble()
