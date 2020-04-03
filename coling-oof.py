#!/usr/bin/env python
# -*- coding: utf-8 -*-
##

import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as VS
from keras.layers import CuDNNGRU, concatenate
from xgboost import XGBClassifier
from sklearn.naive_bayes import MultinomialNB
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.tokenize import RegexpTokenizer
from sklearn.model_selection import KFold

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import normalize, MaxAbsScaler

from fastText import load_model
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix

from keras.layers import Bidirectional, Input, GRU, Dense, Dropout, SpatialDropout1D, GlobalMaxPool1D, GlobalAveragePooling1D, Lambda, Concatenate
from keras.models import Model

import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

from math import log
# Build a cost dictionary, assuming Zipf's law and cost = -math.log(probability).
import sys

if sys.version_info.major == 3:
    unicode = str

frequency_file = "wordsSortedByFrequency.txt"

# following an idea of 
# https://stackoverflow.com/questions/8870261/how-to-split-text-without-spaces-into-list-of-words
# and 
# http://pasted.co/c1666a6b
words = open(frequency_file).read().split()
wordcost = dict((k, log((i+1)*log(len(words)))) for i,k in enumerate(words))
maxword = max(len(x) for x in words)

def infer_spaces(s):
    """Uses dynamic programming to infer the location of spaces in a string
    without spaces."""

    # Find the best match for the i first characters, assuming cost has
    # been built for the i-1 first characters.
    # Returns a pair (match_cost, match_length).
    def best_match(i):
        candidates = enumerate(reversed(cost[max(0, i-maxword):i]))
        return min((c + wordcost.get(s[i-k-1:i], 9e999), k+1) for k,c in candidates)

    # Build the cost array.
    cost = [0]
    for i in range(1,len(s)+1):
        c,k = best_match(i)
        cost.append(c)

    # Backtrack to recover the minimal-cost string.
    out = []
    i = len(s)
    while i>0:
        c,k = best_match(i)
        assert c == cost[i]
        out.append(s[i-k:i])
        i -= k

    return " ".join(reversed(out))

def get_session(gpu_fraction=0.3):
   '''Assume that you have 6GB of GPU memory and want to allocate ~2GB'''

   num_threads = os.environ.get('OMP_NUM_THREADS')
   gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

   if num_threads:
       return tf.Session(config=tf.ConfigProto(
           gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
   else:
       return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

KTF.set_session(get_session())

print('\nLoading FT model')
path = ''
model_path = 'ft_model.bin'
ft_model = load_model(model_path)
n_features = ft_model.get_dimension()
classes = ['OAG','NAG','CAG']
window_length = 150


def str_normalize(s):
    """
    Given a text, cleans and normalizes it. Feel free to add your own stuff.
    """
    s = s.lower()
    s = re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', ' _ip_ ', s)
    s = s.replace('...', ' dots ')
    s = s.replace('..', ' dots ')
    s = re.sub(r'([\'\"\.\(\)\!\?\-\\\/\,])', r' \1 ', s)
    s = re.sub(r'([\;\:\|•«\n])', ' ', s)
    s = s.replace('$', ' $ ')
    s = s.replace('&', ' and ')
    s = s.replace('0', ' zero ')
    s = s.replace('1', ' one ')
    s = s.replace('2', ' two ')
    s = s.replace('3', ' three ')
    s = s.replace('4', ' four ')
    s = s.replace('5', ' five ')
    s = s.replace('6', ' six ')
    s = s.replace('7', ' seven ')
    s = s.replace('8', ' eight ')
    s = s.replace('9', ' nine ')
    return s


def text_to_vector(text, embedding=True):
    """
    Given a string, normalizes it, then splits it into words and finally converts
    it to a sequence of word vectors.
    """
    text = str_normalize(text)
    words = text.split()
    words_without_hashtags = []
    for word in words:
        if word.startswith('#'):
            word_without_hashtag = infer_spaces(word[1:].replace('_',''))
            words_without_hashtags += ['hashtag'] + word_without_hashtag.split()
        else:
            if word.startswith('@'):
                word_without_hashtag = infer_spaces(word[1:].replace('_',''))
                words_without_hashtags += ['usertag'] + word_without_hashtag.split()
            else:
                words_without_hashtags += [word]
    words = words_without_hashtags
    if not embedding:
        one_string = ' '.join(words)
        if sys.version_info.major == 3:
            return one_string
        return unicode(one_string, errors='ignore')

    window = words[-window_length:]

    x = np.zeros((window_length, n_features))

    for i, word in enumerate(window):
        x[i, :] = ft_model.get_word_vector(word).astype('float32')

    return x


def df_to_data(df):
    """
    Convert a given dataframe to a dataset of inputs for the NN.
    """
    x = np.zeros((len(df), window_length, n_features), dtype='float32')

    for i, comment in enumerate(df['comment_text'].values):
        x[i, :] = text_to_vector(comment)

    return x


def data_generator(df, batch_size):
    """
    Given a raw dataframe, generates infinite batches of FastText vectors.
    """
    batch_i = 0  # Counter inside the current batch vector
    batch_x = None  # The current batch's x data
    batch_y = None  # The current batch's y data

    while True:  # Loop forever
        df = df.sample(frac=1)  # Shuffle df each epoch

        for i, row in df.iterrows():
            comment = row['comment_text']

            if batch_x is None:
                batch_x = np.zeros((batch_size, window_length, n_features), dtype='float32')
                batch_y = np.zeros((batch_size, len(classes)), dtype='float32')

            batch_x[batch_i] = text_to_vector(comment)
            batch_y[batch_i] = row[classes].values
            batch_i += 1

            if batch_i == batch_size:
                yield batch_x, batch_y
                batch_x = None
                batch_y = None
                batch_i = 0


def read_data(full_path, header=None):
    df = pd.read_csv(filepath_or_buffer=full_path, header=header, names=['comment_text', 'label'])
    df['OAG'] = np.where(df['label'] == 'OAG', 1, 0)
    df['NAG'] = np.where(df['label'] == 'NAG', 1, 0)
    df['CAG'] = np.where(df['label'] == 'CAG', 1, 0)
    df = df.drop(['label'], axis=1)
    return df


def read_coling_data():
    train_file = "./english/agr_en_train.csv"
    train_file_translations = []#["./english/extended_data/train_de.csv","./english/extended_data/dev_de.csv","./english/extended_data/dev_es.csv","./english/extended_data/train_es.csv"] #["./english/extended_data/train/train_de.csv"], "./english/extended_data/train/train_es.csv", "./english/extended_data/train/train_fr.csv"] + ["./hindi/extended_data/train/train_en.csv"] #"./hindi/extended_data/train/train_de.csv", "./hindi/extended_data/train/train_es.csv", "./hindi/extended_data/train/train_fr.csv",
    dev_file = "./english/agr_en_dev.csv"

    df_train = read_data(path+train_file)
    df_dev = read_data(path+dev_file)

    for train_file_translated in train_file_translations:
        df_train_translated = read_data(path+train_file_translated, header=0)
        df_train = pd.concat([df_train, df_train_translated])
    return df_train, df_dev

def read_coling_data_de(lang='de'):
    train_file = "./english/extended_data/train_"+lang+".csv"
    dev_file = "./english/extended_data/dev_"+lang+".csv"

    df_train = read_data(path+train_file)
    df_dev = read_data(path+dev_file)

    return df_train, df_dev


def read_test_data(full_path, header=None):
    df = pd.read_csv(filepath_or_buffer=full_path, header=header, names=['id', 'comment_text'])
    return df


def read_val_data(full_path, header=None):
    df = pd.read_csv(filepath_or_buffer=full_path, header=header, names=['id', 'comment_text', 'label'])
    df = df.drop(['label'], axis=1)
    return df

def model5(top_k=2, num_filters=64):
    def _top_k(x):
        x = tf.transpose(x, [0, 2, 1])
        k_max = tf.nn.top_k(x, k=top_k)
        return tf.reshape(k_max[0], (-1, 2 * num_filters * top_k))

    inp = Input(shape=(window_length, n_features))
    x = SpatialDropout1D(0.1)(x)
    x = Bidirectional(CuDNNGRU(num_filters, return_sequences=True))(inp)
    avg_pool = GlobalAveragePooling1D()(x)
    k_max = Lambda(_top_k)(x)
    conc = concatenate([avg_pool, k_max])
    conc = Dropout(0.1)(conc)
    x = Dense(len(classes), activation="softmax")(conc)
    return Model(inputs=inp, outputs=x)

def eval(y_val, y_val_pred):
    y_val_pred = np.argmax(y_val_pred, axis=1)
    y_val_pred = np.asarray([[1,0,0] if x == 0 else [0,1,0] if x == 1 else [0,0,1] for x in y_val_pred])
    score = f1_score(y_val, y_val_pred, average='weighted')
    print(score)
    print(classification_report(y_val,y_val_pred, target_names=classes))
    print(confusion_matrix(y_val.argmax(axis=1),y_val_pred.argmax(axis=1)))
    return score


def tfidf(df_train, df_dev, df_all):
    x_train = df_train['comment_text'].apply(lambda x: text_to_vector(x, embedding=False)).values
    y_train = df_train[classes].values
    x_val = df_dev['comment_text'].apply(lambda x: text_to_vector(x, embedding=False)).values
    vectorizer = TfidfVectorizer(min_df=2, ngram_range=(1, 2), smooth_idf=True, max_df=0.5, stop_words='english')
    x_all = df_all['comment_text'].apply(lambda x: text_to_vector(x, embedding=False)).values
    vectorizer.fit(x_all)

    nb = OneVsRestClassifier(LogisticRegression(solver='sag', class_weight="balanced"))
    x_train = vectorizer.transform(x_train)
    y_train = np.argmax(y_train, axis=1)
    nb.fit(x_train, y_train)
    x_val = vectorizer.transform(x_val)
    y_val_pred = nb.predict_proba(x_val)
    print(nb.multilabel_)
    return y_val_pred, vectorizer, nb


def rnn(df_train, df_dev):
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    batch_size = 32
    training_steps_per_epoch = round(len(df_train) / batch_size)
    training_generator = data_generator(df_train, batch_size)

    print('\nFitting model')
    x_val = df_to_data(df_dev)
    y_val = df_dev[classes].values
    model.fit_generator(training_generator, steps_per_epoch=training_steps_per_epoch, epochs=2, callbacks=[], verbose=1,
                        validation_data=(x_val, y_val))
    y_val_pred = model.predict(x_val, verbose=1, batch_size=32)
    return y_val_pred



pos_code_map = {'CC': 'A', 'CD': 'B', 'DT': 'C', 'EX': 'D', 'FW': 'E', 'IN': 'F', 'JJ': 'G', 'JJR': 'H', 'JJS': 'I',
                'LS': 'J', 'MD': 'K', 'NN': 'L', 'NNS': 'M',
                'NNP': 'N', 'NNPS': 'O', 'PDT': 'P', 'POS': 'Q', 'PRP': 'R', 'PRP$': 'S', 'RB': 'T', 'RBR': 'U',
                'RBS': 'V', 'RP': 'W', 'SYM': 'X', 'TO': 'Y', 'UH': 'Z',
                'VB': '1', 'VBD': '2', 'VBG': '3', 'VBN': '4', 'VBP': '5', 'VBZ': '6', 'WDT': '7', 'WP': '8',
                'WP$': '9', 'WRB': '@'}

code_pos_map = {v: k for k, v in pos_code_map.items()}


# abbreviation converters
def convert(tag):
    try:
        code = pos_code_map[tag]
    except:
        code = '?'
    return code


def inv_convert(code):
    try:
        tag = code_pos_map[code]
    except:
        tag = '?'
    return tag


def pos_tags(text):
    tokenizer = RegexpTokenizer(r'\w+')
    text_processed = tokenizer.tokenize(text)
    return "".join(convert(tag) for (word, tag) in nltk.pos_tag(text_processed))


def text_pos_inv_convert(text):
    return "-".join(inv_convert(c.upper()) for c in text)


def pos_tag_approach(df_train, df_dev):
    print('fitting pos tag features')
    pos_vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        strip_accents='unicode',
        analyzer='char',
        ngram_range=(3, 3),
        max_features=200)
    x_train = df_train['comment_text'].apply(lambda x: pos_tags(text_to_vector(x, embedding=False))).values
    x_val = df_dev['comment_text'].apply(lambda x: pos_tags(text_to_vector(x, embedding=False))).values
    y_train = df_train[classes].values
    y_val = df_dev[classes].values
    pos_vectorizer.fit(x_train) # todo concat x_val

    nb = OneVsRestClassifier(LogisticRegression(solver='sag'))
    x_train = pos_vectorizer.transform(x_train)
    nb.fit(x_train, y_train)
    x_val = pos_vectorizer.transform(x_val)
    y_val_pred = nb.predict(x_val)
    return y_val_pred


def char_approach(df_train, df_dev, df_all):
    print('fitting char features')
    char_vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        strip_accents='unicode',
        analyzer='char',
        stop_words='english',
        ngram_range=(2, 6),
        max_features=50000)
    x_train = df_train['comment_text'].apply(lambda x: text_to_vector(x, embedding=False)).values
    x_val = df_dev['comment_text'].apply(lambda x: text_to_vector(x, embedding=False)).values
    y_train = df_train[classes].values
    x_all = df_all['comment_text'].apply(lambda x: text_to_vector(x, embedding=False)).values
    char_vectorizer.fit(x_all)

    nb = OneVsRestClassifier(LogisticRegression(solver='sag', class_weight="balanced"))
    x_train = char_vectorizer.transform(x_train)
    y_train = np.argmax(y_train, axis=1)
    nb.fit(x_train, y_train)
    x_val = char_vectorizer.transform(x_val)
    y_val_pred = nb.predict_proba(x_val)
    print(nb.multilabel_)
    return y_val_pred, char_vectorizer, nb


def predict_for_test_set(model, df_test):
    x_test = df_to_data(df_test)
    return model.predict(x_test, verbose=1, batch_size=32)


# emojis:
eyes = "[8:=;]"
nose = "['`\-]?"
smile1 = re.compile(eyes + nose + "[Dd)]")
smile2 = re.compile("[(d]" + nose + eyes)
sad2 = re.compile(eyes + nose + "\(")
sad1 = re.compile("\)" + nose + eyes)
lol = re.compile(eyes + nose + "p")
neutralFace = re.compile(eyes + nose + "[/|l*]")


def count_regexp_occ(regexp="", text=None):
    """ Simple way to get the number of occurence of a regex"""
    if len(text) == 0:
        return 0
    else:
        return len(re.findall(regexp, text)) / float(len(text))

sentiment_analyzer = VS()


def logistic_preprocess(df):
    num_repeated_mark = df['comment_text'].apply(lambda x: x.count('!')).values
    num_repeated_mark = num_repeated_mark.reshape(-1, 1)
    ratio_capitalized = df['comment_text'].apply(lambda x: sum(1 for c in x if c.isupper()) / len(x)) # ratio_capitalized
    ratio_capitalized = ratio_capitalized.reshape(-1, 1)
    num_repeated_question = df['comment_text'].apply(lambda x: x.count('?')).values
    num_repeated_question = num_repeated_question.reshape(-1, 1)
    num_of_words = df['comment_text'].apply(lambda x: len(x.split())) #num_of_words
    num_of_words = num_of_words.reshape(-1, 1)
    num_dot = df['comment_text'].apply(lambda x: x.count('.'))
    num_dot = num_dot.reshape(-1, 1)
    avg_length = df['comment_text'].apply(lambda x: np.average([len(a) for a in x.split()]))
    avg_length = avg_length.reshape(-1, 1)
    num_negations = df['comment_text'].apply(
        lambda x: x.count('not') + x.count('never') + x.count('nothing') + x.count('no'))
    num_negations = num_negations.reshape(-1, 1)
    num_repeated_dot = df['comment_text'].apply(lambda x: x.count('..'))
    num_repeated_dot = num_repeated_dot.reshape(-1, 1)

    count_smile1 = df["comment_text"].apply(lambda x: count_regexp_occ(smile1, x))
    count_smile1 = count_smile1.reshape(-1, 1)
    count_smile2 = df["comment_text"].apply(lambda x: count_regexp_occ(smile2, x))
    count_smile2 = count_smile2.reshape(-1, 1)
    count_sad2 = df["comment_text"].apply(lambda x: count_regexp_occ(sad2, x))
    count_sad2 = count_sad2.reshape(-1, 1)
    count_sad1 = df["comment_text"].apply(lambda x: count_regexp_occ(sad1, x))
    count_sad1 = count_sad1.reshape(-1, 1)
    count_lol = df["comment_text"].apply(lambda x: count_regexp_occ(lol, x))
    count_lol = count_lol.reshape(-1, 1)
    count_neutralFace = df["comment_text"].apply(lambda x: count_regexp_occ(neutralFace, x))
    count_neutralFace = count_neutralFace.reshape(-1, 1)
    sentiment = df['comment_text'].apply(lambda x: sentiment_analyzer.polarity_scores(x)['compound'])
    sentiment = sentiment.reshape(-1, 1)
    num_http = df['comment_text'].apply(lambda x: x.count('http'))
    num_http = num_http.reshape(-1, 1)
    num_comma = df['comment_text'].apply(lambda x: x.count(','))
    num_comma = num_comma.reshape(-1, 1)
    num_heart = df['comment_text'].apply(lambda x: x.count('💚')+x.count('💙')+x.count('💛')+x.count('💜')+x.count('<3')+x.count('😍')+x.count('😘'))
    num_heart = num_heart.reshape(-1, 1)
    num_laugh = df['comment_text'].apply(lambda x: x.count('😀')+x.count('😂')+x.count('😅'))
    num_laugh = num_laugh.reshape(-1, 1)
    num_tongue = df['comment_text'].apply(lambda x: x.count('😝')+x.count('😜')+x.count('😛'))
    num_tongue = num_tongue.reshape(-1, 1)
    num_bot = df['comment_text'].apply(lambda x: x.count('bot')+x.count('BOT')+x.count('Bot'))
    num_bot = num_bot.reshape(-1, 1)
    num_muslim = df['comment_text'].apply(lambda x: x.count('muslim') + x.count('Muslim') + x.count('MUSLIM'))
    num_muslim = num_muslim.reshape(-1, 1)
    num_whatsapp = df['comment_text'].apply(lambda x: x.count('whatsapp') + x.count('Whatsapp') + x.count('WhatsApp'))
    num_whatsapp = num_whatsapp.reshape(-1, 1)
    num_star = df['comment_text'].apply(lambda x: x.count('*'))
    num_star = num_star.reshape(-1, 1)
    num_botsign = df['comment_text'].apply(lambda x: x.count('❌') + x.count('💢'))
    num_botsign = num_botsign.reshape(-1, 1)
    num_angry = df['comment_text'].apply(lambda x: x.count('😡') + x.count('😠'))
    num_angry = num_angry.reshape(-1, 1)
    num_blink = df['comment_text'].apply(lambda x: x.count('😉') + x.count('😏'))
    num_blink = num_blink.reshape(-1, 1)
    num_thumb = df['comment_text'].apply(lambda x: x.count('👍'))
    num_thumb = num_thumb.reshape(-1, 1)
    num_cool = df['comment_text'].apply(lambda x: x.count('😎'))
    num_cool = num_cool.reshape(-1, 1)
    num_pray = df['comment_text'].apply(lambda x: x.count('🙏'))
    num_pray = num_pray.reshape(-1, 1)
    num_clap = df['comment_text'].apply(lambda x: x.count('👏'))
    num_clap = num_clap.reshape(-1, 1)
    num_flush = df['comment_text'].apply(lambda x: x.count('😊'))
    num_flush = num_flush.reshape(-1, 1)
    num_rose = df['comment_text'].apply(lambda x: x.count('🌹'))
    num_rose = num_rose.reshape(-1, 1)
    num_think = df['comment_text'].apply(lambda x: x.count('🤔'))
    num_think = num_think.reshape(-1, 1)
    num_surprise = df['comment_text'].apply(lambda x: x.count('😯'))
    num_surprise = num_surprise.reshape(-1, 1)
    num_exact = df['comment_text'].apply(lambda x: x.count('👌'))
    num_exact = num_exact.reshape(-1, 1)
    num_sad = df['comment_text'].apply(lambda x: x.count('😪'))
    num_sad = num_sad.reshape(-1, 1)

    x = np.concatenate((num_of_words, avg_length, ratio_capitalized, num_repeated_mark, num_repeated_question, num_dot, num_negations, num_repeated_dot, count_smile1, count_smile2, count_lol, count_sad1, count_sad2, count_neutralFace, sentiment, num_http, num_comma, num_botsign, num_bot, num_heart, num_laugh, num_muslim, num_star, num_whatsapp, num_tongue, num_angry, num_blink,num_thumb, num_sad, num_cool, num_blink, num_clap, num_exact, num_surprise, num_think, num_flush, num_pray, num_rose), axis=1)
    return x


def logistic_scale(x, scaler):
    x = scaler.transform(x)
    x = normalize(x, norm='l2', axis=0, copy=False)
    return x


def logistic_regression(df_train, df_dev):
    x_train = logistic_preprocess(df_train)
    x_val = logistic_preprocess(df_dev)

    scaler = MaxAbsScaler()
    scaler.fit(x_train)
    x_train = logistic_scale(x_train, scaler)
    x_val = logistic_scale(x_val, scaler)
    y_train = df_train[classes].values

    nb = OneVsRestClassifier(LogisticRegression(solver='sag', class_weight="balanced"))
    y_train = np.argmax(y_train, axis=1)
    nb.fit(x_train, y_train)
    print(y_train)
    y_val_pred = nb.predict_proba(x_val)
    print(nb.multilabel_)
    return y_val_pred, nb, scaler

approaches = {'rnn': rnn, 'tfidf': tfidf, 'pos' : pos_tag_approach, 'char': char_approach, 'logreg': logistic_regression}

df_train, df_dev = read_coling_data()
df_train_de, df_dev_de = read_coling_data_de('de')
df_train_fr, df_dev_fr = read_coling_data_de('fr')
df_train_es, df_dev_es = read_coling_data_de('es')
df_test = read_test_data("./test/agr_en_sm_test.csv")#sm = fb
df_all = pd.concat([df_train, df_dev])
df_all_de = pd.concat([df_train_de, df_dev_de])
df_all_fr = pd.concat([df_train_fr, df_dev_fr])
df_all_es = pd.concat([df_train_es, df_dev_es])
n_folds = 10
model_numbers = [1,2,3,4]
avgscore = []
for model_number in model_numbers:
    #model_number = 1 # 1:rnn, 2:tfidf, 3: char
    submission = read_test_data("./test/agr_en_sm_test.csv")#sm = fb
    first = True
    oof_predictions = None
    skfolds = KFold(n_splits=n_folds, random_state=42)
    score = 0
    for train_index, dev_index in skfolds.split(df_all):
        df_train = df_all.iloc[train_index] #add also the translated versions with these indexes
        df_dev = df_all.iloc[dev_index] #stays as it is

        df_train_de = df_all_de.iloc[train_index]
        df_train_fr = df_all_fr.iloc[train_index]
        df_train_es = df_all_es.iloc[train_index]

        df_train = pd.concat([df_train, df_train_de, df_train_fr, df_train_es])
        #df_dev_de = df_all_de.iloc[dev_index]

        submission_part = df_all.iloc[dev_index]

        model = model5()
        y_val = df_dev[classes].values

        if model_number == 1:
            y_val_pred = approaches['rnn'](df_train, df_dev)
        elif model_number == 2:
            y_val_pred, vectorizer, nb = approaches['tfidf'](df_train, df_dev, df_all)
        elif model_number == 3:
            y_val_pred, vectorizer, nb = approaches['char'](df_train, df_dev, df_all)
        elif model_number == 4:
            y_val_pred, nb, scaler = approaches['logreg'](df_train, df_dev)
            print('#')
            print(y_val_pred)

        score += eval(y_val, y_val_pred)

        submission_part['OAG'] = np.asarray([x[0] for x in y_val_pred])
        submission_part['NAG'] = np.asarray([x[1] for x in y_val_pred])
        submission_part['CAG'] = np.asarray([x[2] for x in y_val_pred])

        if oof_predictions is None:
            oof_predictions = submission_part

        else:
            oof_predictions = pd.concat([oof_predictions, submission_part])

        if model_number == 1:
            y_test_pred = predict_for_test_set(model,df_test)
        elif model_number == 2 or model_number == 3:
            x_train = df_test['comment_text'].apply(lambda x: text_to_vector(x, embedding=False)).values
            x_train = vectorizer.transform(x_train)
            y_test_pred = nb.predict_proba(x_train)
        elif model_number == 4:
            x_train = logistic_preprocess(df_test)
            x_train = logistic_scale(x_train, scaler)
            y_test_pred = nb.predict_proba(x_train)


        if first:
            submission['OAG'] = np.asarray([x[0] for x in y_test_pred]) / n_folds
            submission['NAG'] = np.asarray([x[1] for x in y_test_pred]) / n_folds
            submission['CAG'] = np.asarray([x[2] for x in y_test_pred]) / n_folds
            first = False
        else:
            submission['OAG'] += np.asarray([x[0] for x in y_test_pred]) / n_folds
            submission['NAG'] += np.asarray([x[1] for x in y_test_pred]) / n_folds
            submission['CAG'] += np.asarray([x[2] for x in y_test_pred]) / n_folds

        oof_predictions.to_csv(path + "oof" + str(model_number) + ".csv", index=False)
        submission.to_csv(path + "sub" + str(model_number) + ".csv", index=False)
    score /= n_folds
    print(score)
    avgscore += [score]
for score in avgscore:
    print(score)
