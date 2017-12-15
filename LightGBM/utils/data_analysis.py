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

train_columns = ['target', 'bd', 'membership_days', 'song_length', 'genre_count', 'artist_count', 'composer_count',
                 'lyricist_count', 'count_song_played', 'count_artist_played', 'count_genre_played',
                 'count_genre_liked', 'genre_like_ratio', 'listen_count', 'acl_proba', 'user_id', 'song_id',
                 'source_system_tab', 'source_screen_name', 'source_type', 'city', 'gender', 'registered_via',
                 'registration_year', 'registration_month', 'registration_date', 'expiration_year', 'expiration_month',
                 'expiration_date', 'genre_ids', 'artist', 'composer', 'lyricist', 'language', 'smaller_song',
                 'artist_composer', 'artist_composer_lyricist', 'song_country', 'song_publisher', 'song_year']

test_columns = ['id', 'bd', 'membership_days', 'song_length', 'genre_count', 'artist_count', 'composer_count',
                'lyricist_count', 'count_song_played', 'count_artist_played', 'count_genre_played',
                'count_genre_liked', 'genre_like_ratio', 'listen_count', 'acl_proba', 'user_id', 'song_id',
                'source_system_tab', 'source_screen_name', 'source_type', 'city', 'gender', 'registered_via',
                'registration_year', 'registration_month', 'registration_date', 'expiration_year', 'expiration_month',
                'expiration_date', 'genre_ids', 'artist', 'composer', 'lyricist', 'language', 'smaller_song',
                'artist_composer', 'artist_composer_lyricist', 'song_country', 'song_publisher', 'song_year']

train_columns2 = ['target', 'bd', 'membership_days', 'song_length', 'genre_count', 'artist_count', 'composer_count',
                  'lyricist_count', 'count_song_played', 'count_artist_played', 'count_genre_played', 'listen_count',
                  'user_id', 'song_id', 'source_system_tab', 'source_screen_name', 'source_type', 'city', 'gender',
                  'registered_via', 'registration_year', 'registration_month', 'registration_date', 'expiration_year',
                  'expiration_month', 'expiration_date', 'genre_ids', 'artist', 'composer', 'lyricist', 'language',
                  'smaller_song', 'artist_composer', 'artist_composer_lyricist', 'song_country', 'song_publisher',
                  'song_year']

test_columns2 = ['id', 'bd', 'membership_days', 'song_length', 'genre_count', 'artist_count', 'composer_count',
                 'lyricist_count', 'count_song_played', 'count_artist_played', 'count_genre_played', 'listen_count',
                 'user_id', 'song_id', 'source_system_tab', 'source_screen_name', 'source_type', 'city', 'gender',
                 'registered_via', 'registration_year', 'registration_month', 'registration_date', 'expiration_year',
                 'expiration_month', 'expiration_date', 'genre_ids', 'artist', 'composer', 'lyricist', 'language',
                 'smaller_song', 'artist_composer', 'artist_composer_lyricist', 'song_country', 'song_publisher',
                 'song_year']

ffm_columns = {'bd': 'I1', 'membership_days': 'I2', 'song_length': 'I3', 'genre_count': 'I4', 'artist_count': 'I5',
               'composer_count': 'I6', 'lyricist_count': 'I7', 'count_song_played': 'I8', 'count_artist_played': 'I9',
               'count_genre_played': 'I10', 'count_genre_liked': 'I11', 'genre_like_ratio': 'I12',
               'listen_count': 'I13', 'acl_proba': 'I14', 'user_id': 'C1', 'song_id': 'C2', 'source_system_tab': 'C3',
               'source_screen_name': 'C4', 'source_type': 'C5', 'city': 'C6', 'gender': 'C7', 'registered_via': 'C8',
               'registration_year': 'C9', 'registration_month': 'C10', 'registration_date': 'C11',
               'expiration_year': 'C12', 'expiration_month': 'C13', 'expiration_date': 'C14', 'genre_ids': 'C15',
               'artist': 'C16', 'composer': 'C17', 'lyricist': 'C18', 'language': 'C19', 'smaller_song': 'C20',
               'artist_composer': 'C21', 'artist_composer_lyricist': 'C22', 'song_country': 'C23',
               'song_publisher': 'C24', 'song_year': 'C25'}

ffm_columns2 = {'bd': 'I1', 'membership_days': 'I2', 'song_length': 'I3', 'genre_count': 'I4', 'artist_count': 'I5',
                'composer_count': 'I6', 'lyricist_count': 'I7', 'count_song_played': 'I8', 'count_artist_played': 'I9',
                'count_genre_played': 'I10', 'listen_count': 'I11', 'user_id': 'C1', 'song_id': 'C2',
                'source_system_tab': 'C3', 'source_screen_name': 'C4', 'source_type': 'C5', 'city': 'C6',
                'gender': 'C7', 'registered_via': 'C8', 'registration_year': 'C9', 'registration_month': 'C10',
                'registration_date': 'C11', 'expiration_year': 'C12', 'expiration_month': 'C13',
                'expiration_date': 'C14', 'genre_ids': 'C15', 'artist': 'C16', 'composer': 'C17', 'lyricist': 'C18',
                'language': 'C19', 'smaller_song': 'C20', 'artist_composer': 'C21', 'artist_composer_lyricist': 'C22',
                'song_country': 'C23', 'song_publisher': 'C24', 'song_year': 'C25'}


def analysis_file():
    def merge():
        train = pd.read_csv(data_path + 'train.csv')
        test = pd.read_csv(data_path + 'test.csv')
        # songs = pd.read_csv(data_path + 'songs.csv')
        songs = pd.read_csv(data_path + 'songs.fixed.csv')
        # members = pd.read_csv(data_path + 'members.csv')
        # songs_extra = pd.read_csv(data_path + 'song_extra_info.csv')

        train = train.merge(songs, on='song_id', how='left')
        test = test.merge(songs, on='song_id', how='left')
        # train = train.merge(members, on='user_id', how='left')
        # test = test.merge(members, on='user_id', how='left')
        # train = train.merge(songs_extra, on='song_id', how='left')
        # test = test.merge(songs_extra, on='song_id', how='left')

        # train = train[pd.notnull(train.artist)]
        # train = train[train.artist.notnull()]

        # dh.deal_nan_value(train_merged_df=train, test_merged_df=test)
        dh.check_missing_values(train)
        dh.check_missing_values(test)

        # train.to_csv(data_path + 'train_lgbm.csv', index=False, columns=train_columns)
        # test.to_csv(data_path + 'test_lgbm.csv', index=False, columns=test_columns)

    def query():
        train = pd.read_csv(data_path + 'train_lgbm.csv')
        # test = pd.read_csv(data_path + 'test.csv')
        # songs = pd.read_csv(data_path + 'songs.csv')
        # train = train.merge(songs, on='song_id', how='left')
        # query = test[test.genre_like_ratio.isnull()]
        # query = songs[songs.name.isin(['風繼續吹'])]
        # query = songs[songs.name.isin(['花火']) & songs.artist.isin(['汪峰'])]
        # query.to_csv(data_path + 'query.csv', index=False)
        print(train.source_system_tab.unique())
        print(train.source_screen_name.unique())
        print(train.source_type.unique())

    def concat():
        songs = pd.read_csv(data_path + 'songs.csv')
        test_nan = pd.read_csv(data_path + 'test_nan.csv')
        songs = pd.concat([songs, test_nan])
        songs.to_csv(data_path + 'songs_haha.csv', index=False)

    def view(feature):
        train_lgbm = pd.read_csv(data_path + 'train_lgbm_2.csv')
        test_lgbm = pd.read_csv(data_path + 'test_lgbm_2.csv')

        train_lgbm['bd'] = train_lgbm['bd'].apply(dh.deal_bd_info)
        test_lgbm['bd'] = test_lgbm['bd'].apply(dh.deal_bd_info)

        print(train_lgbm[feature].min())
        print(train_lgbm[feature].max())
        print(test_lgbm[feature].min())
        print(test_lgbm[feature].max())

        train_lgbm.to_csv(data_path + 'train_lgbm_3.csv', index=False)
        test_lgbm.to_csv(data_path + 'test_lgbm_3.csv', index=False)

    def select():
        train = pd.read_csv(data_path + 'train_lgbm.csv')
        test = pd.read_csv(data_path + 'test_lgbm.csv')

        # train['bd'] = train['bd'].apply(dh.deal_bd_info)
        # train['source_system_tab'] = train['source_system_tab'].apply(dh.deal_source_system_tab_info)
        # train['source_screen_name'] = train['source_screen_name'].apply(dh.deal_source_screen_name_info)
        # train['source_type'] = train['source_type'].apply(dh.deal_source_type_info)
        # test['bd'] = test['bd'].apply(dh.deal_bd_info)
        # test['source_system_tab'] = test['source_system_tab'].apply(dh.deal_source_system_tab_info)
        # test['source_screen_name'] = test['source_screen_name'].apply(dh.deal_source_screen_name_info)
        # test['source_type'] = test['source_type'].apply(dh.deal_source_type_info)

        # dh.deal_nan_value(train_merged_df=train, test_merged_df=test)
        # dh.check_missing_values(train)
        # dh.check_missing_values(test)
        # train = train[train_columns2]
        # test = test[test_columns2]
        train.rename(columns=ffm_columns2, inplace=True)
        test.rename(columns=ffm_columns2, inplace=True)
        train.to_csv(data_path + 'train_ffm.csv', index=False)
        test.to_csv(data_path + 'test_ffm.csv', index=False)

    def view_target_avg():
        train = pd.read_csv(data_path + 'train.csv')
        avg_target = 0.0
        for item in train['target']:
            avg_target += float(item)
        print(avg_target / len(train['target']))

    # merge()
    # view(feature='bd')
    select()
    # query()
    # view_target_avg()


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


def ensemble(percent_fm=0.2, percent_ffm=0.2, percent_xgboost=0.3):
    lbgm = pd.read_csv(result_path + 'submission_lgbm_best.csv')
    xgboost = pd.read_csv(result_path + 'submission_xgboost_best.csv')
    ffm = pd.read_csv(result_path + 'submission_ffm_best.csv')
    fm = pd.read_csv(result_path + 'submission_fm_best.csv')
    avg = pd.read_csv(result_path + 'submission_final.csv')

    p_test_avg = percent_fm * fm['target'] + percent_ffm * ffm['target'] + percent_xgboost * xgboost['target'] \
                 + (1 - percent_fm - percent_ffm - percent_xgboost) * lbgm['target']

    subm_avg = pd.DataFrame()
    subm_avg['id'] = lbgm['id']
    subm_avg['target'] = p_test_avg
    subm_avg.to_csv(result_path + 'submission_final_fm_ffm_xgboost_lgbm.csv', index=False, float_format='%.5f')


def detect(file):
    with open(file, "rb") as f:
        data = f.read()
    print(chardet.detect(data))


if __name__ == '__main__':
    analysis_file()
    # tune_file()
    # load_pickle_file()
    # ensemble(percent_fm=0.2, percent_ffm=0.1, percent_xgboost=0.3)
