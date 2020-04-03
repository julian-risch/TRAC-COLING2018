#!/usr/bin/env python
# -*- coding: utf-8 -*-
##

import pandas as pd
import numpy as np
import re
import lightgbm as lgb
import warnings
from sklearn.model_selection import KFold
warnings.filterwarnings(action='ignore', category=DeprecationWarning, module='sklearn')
from sklearn.preprocessing import StandardScaler

def get_coefs(word, *arr):
    return word, True


def load_embeddings(embedding_path="/Users/julian/Desktop/glove.840B.300d.txt"):
    return dict(get_coefs(*o.strip().split(" ")) for o in open(embedding_path))


embedding_index = None#load_embeddings()


def has_embedding(word):
    embedding_vector = embedding_index.get(word)
    return embedding_vector is not None


def engineer_feature(series, func, normalize=True):
    feature = series.apply(func)

    if normalize:
        feature = pd.Series(z_normalize(feature.values.reshape(-1, 1)).reshape(-1, ))
    feature.name = func.__name__
    return feature


def engineer_features(series, funclist, normalize=True):
    features = pd.DataFrame()
    for func in funclist:
        feature = engineer_feature(series, func, normalize)
        features[feature.name] = feature
    return features


scaler = StandardScaler()


def z_normalize(data):
    scaler.fit(data)
    return scaler.transform(data)


def asterix_freq(x):
    return x.count('!') / float(len(x))

def uppercase_freq(x):
    return len(re.findall(r'[A-Z]', x)) / float(len(x))

def nonalpha_freq(x):
    return len(re.findall(r'[A-Za-z]', x)) / float(len(x))

def has_embedding_freq(x):
    freq = 0
    for word in x.split():
        if has_embedding(word):
            freq += 1
    return freq / float(len(x.split()))


path = ''
model_path = ''
subnums = [1, 2, 3, 4]

def get_subs(nums):
    subs = np.hstack(
        [np.array(pd.read_csv(model_path+"sub" + str(num) + ".csv", usecols=['OAG','NAG','CAG'])) for num in subnums])

    oofs = np.hstack(
        [np.array(pd.read_csv(model_path+"oof" + str(num) + ".csv", usecols=['OAG','NAG','CAG'])) for num in subnums])
    return subs, oofs

def read_data(full_path, header=None):
    df = pd.read_csv(filepath_or_buffer=full_path, header=header, names=['comment_text', 'label'])
    df['OAG'] = np.where(df['label'] == 'OAG', 1, 0)
    df['NAG'] = np.where(df['label'] == 'NAG', 1, 0)
    df['CAG'] = np.where(df['label'] == 'CAG', 1, 0)
    df = df.drop(['label'], axis=1)
    return df

def read_coling_data():
    train_file = path+"./english/agr_en_train.csv"
    train_file_translations = [] #["./english/extended_data/train/train_de.csv"], "./english/extended_data/train/train_es.csv", "./english/extended_data/train/train_fr.csv"] + ["./hindi/extended_data/train/train_en.csv"] #"./hindi/extended_data/train/train_de.csv", "./hindi/extended_data/train/train_es.csv", "./hindi/extended_data/train/train_fr.csv",
    dev_file = path+"./english/agr_en_dev.csv"

    df_train = read_data(path+train_file)
    df_dev = read_data(path+dev_file)

    for train_file_translated in train_file_translations:
        df_train_translated, df_train_translated2 = read_data(path+train_file_translated, header=0)
        df_train = pd.concat([df_train, df_train_translated])
    return df_train, df_dev

def read_test_data(full_path, header=None):
    df = pd.read_csv(filepath_or_buffer=full_path, header=header, names=['id', 'comment_text'])
    return df

def main():

    df_train, df_dev = read_coling_data()
    test_data_path = "./test/agr_en_sm_test.csv"#sm = fb
    #test_data_path = "agr_en_dev.csv"
    df_test = read_test_data(path+test_data_path)
    submission = read_test_data(path + test_data_path)
    df_all = pd.concat([df_train, df_dev])
    submission = submission.drop('comment_text',axis=1)
    INPUT_COLUMN = "comment_text"

    subs, oofs = get_subs(subnums)

    # Engineer features
    feature_functions = [len, asterix_freq, uppercase_freq, nonalpha_freq]
    features = [f.__name__ for f in feature_functions]
    F_train = engineer_features(df_all[INPUT_COLUMN], feature_functions)
    F_test = engineer_features(df_test[INPUT_COLUMN], feature_functions)

    X_train = np.hstack([F_train[features].as_matrix(), oofs])
    X_test = np.hstack([F_test[features].as_matrix(), subs])

    stacker = lgb.LGBMClassifier(max_depth=3, metric="multi_logloss", n_estimators=75, num_leaves=10, boosting_type="gbdt",
                                 learning_rate=0.1, feature_fraction=0.45, colsample_bytree=0.45, bagging_fraction=0.8,
                                 bagging_freq=5, reg_lambda=0.2)

    # Fit and submit
    classes = ['OAG', 'NAG', 'CAG']
    n_folds = 10

    skfolds = KFold(n_splits=n_folds, random_state=42)
    oof_predictions = None

    first = True
    for train_index, dev_index in skfolds.split(df_all):
        X_train_fold = X_train[train_index]
        train_fold = df_all.iloc[train_index]
        train_fold = train_fold.drop('comment_text',axis=1)
        print(train_fold.values)
        train_fold = np.argmax(train_fold.values, axis=1)
        stacker.fit(X_train_fold, train_fold)

        if first:
            print(stacker.predict_proba(X_test)[:] / n_folds)
            submission['OAG'] = stacker.predict_proba(X_test)[:, 0] / n_folds
            submission['NAG'] = stacker.predict_proba(X_test)[:, 1] / n_folds
            submission['CAG'] = stacker.predict_proba(X_test)[:, 2] / n_folds

            first = False
        else:
            submission['OAG'] += stacker.predict_proba(X_test)[:, 0] / n_folds
            submission['NAG'] += stacker.predict_proba(X_test)[:, 1] / n_folds
            submission['CAG'] += stacker.predict_proba(X_test)[:, 2] / n_folds

    submission.to_csv(path + "ensemble.csv", index=False)
    submission.columns = ['id', 'OAG', 'NAG', 'CAG']
    submission_dropped = submission.drop('id', axis=1)
    submission['label'] = np.asarray(['OAG' if x == 0 else 'NAG' if x == 1 else 'CAG' for x in np.argmax(submission_dropped.values, axis=1)])
    submission = submission.drop(classes, axis=1)
    submission.to_csv(path + "ensemble-submission.csv", index=False, header = False)

    from sklearn.model_selection import cross_val_score
    train_cv = df_all
    train_cv = train_cv.drop('comment_text', axis=1)
    train_cv = np.argmax(train_cv.values, axis=1)
    score = cross_val_score(stacker, X_train, train_cv, cv=n_folds, scoring='f1_weighted')
    print("f1_weighted:", score)

print('start')
main()
