# -*- coding:utf-8 -*-

import joblib
import logging
import numpy as np
import pandas as pd
import scipy.sparse as sp


from itertools import combinations
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import coo_matrix
from lightfm import LightFM

data_path = '../data/'
result_path = '../result/'
logs_path = '../logs/'

date_columns = ['expiration_date', 'registration_init_time']
not_categorical_columns = ['target', 'song_length', 'registration_init_time', 'expiration_date', 'time', 'bd']


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


def preprocess():
    train = pd.read_csv(data_path + "train.csv")
    test = pd.read_csv(data_path + "test.csv", index_col=0)
    song_data = pd.read_csv(data_path + "songs.csv")
    user_data = pd.read_csv(data_path + "members.csv", parse_dates=date_columns)

    all_data = pd.concat([train, test])
    all_data = all_data.merge(song_data, on='song_id', how='left')
    all_data = all_data.merge(user_data, on='msno', how='left')

    enc = LabelEncoder()

    for col in [
        'msno', 'song_id', 'source_screen_name', 'source_system_tab', 'source_type',
        'genre_ids', 'artist_name', 'composer', 'lyricist', 'gender'
    ]:
        all_data[col] = enc.fit_transform(all_data[col].fillna('nan'))

    for col in ['language', 'city', 'registered_via']:
        all_data[col] = enc.fit_transform(all_data[col].fillna(-2))

    all_data['time'] = all_data.index / len(all_data)

    n = len(train)
    train_data = all_data[:n]
    test_data = all_data[n:]

    train_data.to_hdf(data_path + 'train.hdf', key='wsdm')
    test_data.to_hdf(data_path + 'test.hdf', key='wsdm')


def create_features():
    train_data = pd.read_hdf(data_path + 'train.hdf', parse_dates=date_columns)
    test_data = pd.read_hdf(data_path + 'test.hdf', parse_dates=date_columns)

    all_data = pd.concat([train_data, test_data])
    df_test = test_data
    df_history_test = train_data

    df_trains = []
    df_history_trains = []

    n = len(test_data)
    shift = int(0.05 * len(train_data))

    for i in range(2):
        m = - (i * shift)
        if m == 0:
            m = None
        df_trains.append(train_data[-(n + i * shift):m])
        df_history_trains.append(train_data[:-(n + i * shift)])

    categorical_columns = all_data.columns.difference(not_categorical_columns)

    orders = {}
    for col in categorical_columns:
        orders[col] = 10 ** (int(np.log(all_data[col].max() + 1) / np.log(10)) + 1)

    print('Orders computing finished...')

    # =================================================
    def get_group(df, cols):
        group = df[cols[0]].copy()
        for col in cols[1:]:
            group = group * orders[col] + df[col]
        return group

    def mean(df, df_history, cols):
        group = get_group(df, cols)
        group_history = get_group(df_history, cols)

        mean_map = df_history.groupby(group_history)['target'].mean()

        return group.map(mean_map).fillna(-1)

    def count(all_data, df, cols):
        group = get_group(df, cols)
        group_all = get_group(all_data, cols)

        count_map = group_all.value_counts()
        return group.map(count_map).fillna(0)

    def regression(df, df_history, cols):
        group = get_group(df, cols)
        group_history = get_group(df_history, cols)

        targets = {}
        times = {}
        for (y, t), u in zip(df_history[['target', 'time']].values, group_history):
            if u not in targets:
                targets[u] = [y]
                times[u] = [t]
            else:
                targets[u].append(y)
                times[u].append(t)

        linal_user = {}
        for u in times:
            if len(times[u]) > 1:
                A = np.vstack([times[u], np.ones(len(times[u]))]).T
                linal_user[u] = np.linalg.inv(A.T.dot(A)).dot(A.T).dot(targets[u])

        result = []

        for t, u in zip(df['time'], group):
            if u not in times:
                result.append(0.5)
            else:
                if len(times[u]) < 2:
                    result.append(0.5)
                else:
                    result.append(linal_user[u].dot([t, 1]))

        return result

    def time_from_prev_heard(df, df_history, cols):
        group = get_group(df, cols)
        group_history = get_group(df_history, cols)

        last_heard = df_history.groupby(group_history)['time'].last().to_dict()

        result = []

        for t, g in zip(df.time, group):
            if g in last_heard:
                result.append(t - last_heard[g])
            else:
                result.append(-1)
            last_heard[g] = t
        return result

    def time_to_next_heard(df, df_history, cols):
        result = []
        df_reverse = df.sort_index(ascending=False)
        group = get_group(df_reverse, cols)

        next_heard = {}

        for g, t in zip(group, df_reverse['time']):
            if g in next_heard:
                result.append(t - next_heard[g])
            else:
                result.append(-1)
            next_heard[g] = t

        result.reverse()
        return result

    def count_from_future(df, df_history, cols):
        result = []
        df_reverse = df.sort_index(ascending=False)
        group = get_group(df_reverse, cols)

        count = {}
        for g in group.values:
            if g in count:
                result.append(count[g])
                count[g] += 1
            else:
                result.append(0)
                count[g] = 1
        result.reverse()
        return result

    def last_time_diff(df, df_history, cols):
        group = get_group(df, cols)
        last_time = df.groupby(group)['time'].last()
        return group.map(last_time) - df.time

    def count_from_past(df, df_history, cols):
        group = get_group(df, cols)
        count = {}
        result = []
        for g in group.values:
            if g not in count:
                count[g] = 0
            else:
                count[g] += 1
            result.append(count[g])
        return result

    def part_of_unique_song(df):
        group = get_group(all_data, ['msno', 'artist_name'])
        group_df = get_group(df, ['msno', 'artist_name'])

        num_song_by_artist = all_data.groupby('artist_name')['song_id'].nunique()
        num_song_by_user_artist = all_data.groupby(group)['song_id'].nunique()

        s1 = df['artist_name'].map(num_song_by_artist)
        s2 = group_df.map(num_song_by_user_artist)

        return s2 / s1

    def matrix_factorization(df, df_history):
        cols = ['msno', 'source_type']
        group = get_group(df, cols)
        group_history = get_group(df_history, cols)

        encoder = LabelEncoder()
        encoder.fit(pd.concat([group, group_history]))

        df['user_id'] = encoder.transform(group)
        df_history['uesr_id'] = encoder.transform(group_history)

        num_users = max(df['user_id'].max(), df_history['user_id'].max()) + 1
        num_items = max(df['song_id'].max(), df_history['song_id'].max()) + 1
        num_msno = max(df['msno'].max(), df_history['msno'].max()) + 1

        M = coo_matrix(
            (df_history['target'], (df_history['user_id'], df_history['song_id'])),
            shape=(num_users, num_items)
        )

        user_features = pd.concat([df, df_history])[['msno', 'user_id']].drop_duplicates()

        user_features = coo_matrix(
            (np.ones(len(user_features)), (user_features['user_id'], user_features['msno'])),
            shape=[num_users, num_msno]
        )

        user_features = sp.hstack([sp.eye(num_users), user_features])

        model = LightFM(no_components=50, learning_rate=0.1)

        model.fit(
            M,
            epochs=2,
            num_threads=50,
            user_features=user_features
        )

        result = model.predict(
            df['user_id'].values,
            df['song_id'].values,
            user_features=user_features
        )

        return result

    def col_name(cols, func):
        return '_'.join(cols) + '_' + func.__name__

    def feature_engineer(df, df_history):
        X = pd.DataFrame()

        print('Feature_Engineer Begins...')
        print('Totally 6 steps.')

        for num_col in [1, 2]:
            for cols in combinations(categorical_columns, num_col):
                for func in [mean, count, time_to_next_heard, count_from_future,
                             last_time_diff, count_from_past]:
                    print(col_name(cols, func) + ' Begins!')
                    X[col_name(cols, func)] = func(df, df_history, list(cols))
                    print(col_name(cols, func) + ' Finished!')

        print('Step 1 Finished! [1/6]')

        for cols in combinations(categorical_columns, 3):
            for func in [mean, count]:
                X[col_name(cols, func)] = func(df, df_history, list(cols))
            if 'msno' in cols:
                for func in [time_to_next_heard, last_time_diff, count_from_past]:
                    X[col_name(cols, func)] = func(df, df_history, list(cols))

        print('Step 2 Finished! [2/6]')

        for cols in [
            ['msno'],
            ['msno', 'source_type'],
            ['msno', 'genre_ids'],
            ['msno', 'artist_name'],
            ['msno', 'composer'],
            ['msno', 'language'],
            ['song_id']
        ]:
            X[col_name(cols, regression)] = regression(df, df_history, cols)

        print('Step 3 Finished! [3/6]')

        for cols in [
            ['msno'],
            ['msno', 'genre_ids'],
            ['msno', 'composer'],
            ['msno', 'language'],
            ['msno', 'artist_name']
        ]:
            X[col_name(cols, time_from_prev_heard)] = time_from_prev_heard(df, df_history, cols)

        print('Step 4 Finished! [4/6]')

        for col in ['song_length', 'bd']:
            X[col] = df[col]

        for col in ['expiration_date', 'registration_init_time']:
            X[col] = df[col].apply(lambda x: x.toordinal())

        X['part_song_listened'] = df['song_length'] / X['msno_time_to_next_heard']
        X['time_from_test_period'] = np.arange(len(df))
        X['part_of_unique_song'] = part_of_unique_song(df)

        X['matrix_factorization'] = matrix_factorization(df, df_history)

        print('Step 5 Finished! [5/6]')

        for i in [500000, 2000000]:
            for cols in [
                ['msno'],
                ['msno', 'source_type'],
                ['msno', 'genre_ids'],
                ['msno', 'artist_name'],
                ['msno', 'composer'],
                ['msno', 'language'],
                ['song_id']
            ]:
                X[col_name(cols, mean) + str(i)] = mean(df_history[-i:], df, cols)

        print('Step 6 Finished! [6/6]')

        return X

    Xtest = feature_engineer(df_test, df_history_test)
    Xtrain0 = feature_engineer(df_trains[0], df_history_trains[0])
    Xtrain1 = feature_engineer(df_trains[1], df_history_trains[1])

    print('Feature Engineer Finished!')

    Xtest.to_hdf(data_path + 'Xtest.hdf', key='abc')
    Xtrain0.to_hdf(data_path + 'Xtrain0.hdf', key='abc')
    Xtrain1.to_hdf(data_path + 'Xtrain1.hdf', key='abc')

    df_trains[0]['target'].to_hdf(data_path + 'ytrain0.hdf', key='abc')
    df_trains[1]['target'].to_hdf(data_path + 'ytrain1.hdf', key='abc')


def train_model(model_xgb, model_cb):
    Xtrain0 = pd.read_hdf(data_path + 'Xtrain0.hdf')
    ytrain0 = pd.read_hdf(data_path + 'ytrain0.hdf')
    Xtrain1 = pd.read_hdf(data_path + 'Xtrain1.hdf')
    ytrain1 = pd.read_hdf(data_path + 'ytrain1.hdf')
    Xtest = pd.read_hdf(data_path + 'Xtest.hdf')

    # Running XGBoost Model
    model_xgb.fit(Xtrain0, ytrain0)
    p = model_xgb.predict_proba(Xtest)[:, 1]
    joblib.dump(p, data_path + 'p0_xgb_mf')

    model_xgb.fit(Xtrain0.drop('matrix_factorization', axis=1), ytrain0)
    p = model_xgb.predict_proba(Xtest.drop('matrix_factorization', axis=1))[:, 1]
    joblib.dump(p, data_path + 'p0_xgb')

    model_xgb.fit(Xtrain1, ytrain1)
    p = model_xgb.predict_proba(Xtest)[:, 1]
    joblib.dump(p, data_path + 'p1_xgb_mf')

    model_xgb.fit(Xtrain1.drop('matrix_factorization', axis=1), ytrain1)
    p = model_xgb.predict_proba(Xtest.drop('matrix_factorization', axis=1))[:, 1]
    joblib.dump(p, data_path + 'p1_xgb')

    # Running CatBoost Model
    model_cb.fit(Xtrain0, ytrain0)
    p = model_cb.predict_proba(Xtest)[:, 1]
    joblib.dump(p, data_path + 'p0_cb_mf')

    model_cb.fit(Xtrain0.drop('matrix_factorization', axis=1), ytrain0)
    p = model_cb.predict_proba(Xtest.drop('matrix_factorization', axis=1))[:, 1]
    joblib.dump(p, data_path + 'p0_cb')

    model_cb.fit(Xtrain1, ytrain1)
    p = model_cb.predict_proba(Xtest)[:, 1]
    joblib.dump(p, data_path + 'p1_cb_mf')

    model_cb.fit(Xtrain1.drop('matrix_factorization', axis=1), ytrain1)
    p = model_cb.predict_proba(Xtest.drop('matrix_factorization', axis=1))[:, 1]
    joblib.dump(p, data_path + 'p1_cb')


def blend(pct1, pct2):
    p0_xgb_mf = joblib.load(data_path + 'p0_xgb_mf')
    p0_xgb = joblib.load(data_path + 'p0_xgb')
    p1_xgb_mf = joblib.load(data_path + 'p1_xgb_mf')
    p1_xgb = joblib.load(data_path + 'p1_xgb')

    p0_cb_mf = joblib.load(data_path + 'p0_cb_mf')
    p0_cb = joblib.load(data_path + 'p0_cb')
    p1_cb_mf = joblib.load(data_path + 'p1_cb_mf')
    p1_cb = joblib.load(data_path + 'p1_cb')

    p_xgb = pct1 * p0_xgb + pct2 * p1_xgb
    p_xgb_mf = pct1 * p0_xgb_mf + pct2 * p1_xgb_mf
    p_cb = pct1 * p0_cb + pct2 * p1_cb
    p_cb_mf = pct1 * p0_cb_mf + pct2 * p1_cb_mf

    p_x = pct1 * p_xgb_mf + pct2 * p_xgb
    p_c = pct1 * p_cb_mf + pct2 * p_cb

    p = pct1 * p_c + pct2 * p_x

    return p


def make_submission(data):
    sub = pd.DataFrame(data)
    sub = sub.reset_index()
    sub.columns = ['id', 'target']
    sub.to_csv(result_path + 'submission.csv', index=False)


if __name__ == '__main__':
    create_features()
