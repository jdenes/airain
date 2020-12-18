import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import tensorflow as tf

from transformers import TFRobertaForSequenceClassification, RobertaTokenizer
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from sklearn import metrics

from data_preparation import KEYWORDS
from utils.basics import clean_string
from utils.plots import nice_plot
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

MODEL_PATH = "../models/tf-roberta/"
T0, T1, T2 = '2000-01-01', '2020-04-01', '2020-08-01'

# tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
# model = TFRobertaForSequenceClassification.from_pretrained('roberta-base')
model = TFAutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')


def balance(df):
    g = df.groupby('labels')
    df = g.apply(lambda x: x.sample(g.size().min(), random_state=123))
    df.index = df.index.droplevel(0)
    return df


def good_news_index(date):

    def next_day(d):
        if d.weekday() == 4:
            return d + timedelta(days=3)
        elif d.weekday() == 5:
            return d + timedelta(days=2)
        else:
            return d + timedelta(days=1)

    def previous_day(d):
        if d.weekday() == 0:
            return d - timedelta(days=3)
        elif d.weekday() == 6:
            return d - timedelta(days=2)
        else:
            return d - timedelta(days=1)

    # Weekend: pretend it was published monday morning
    if date.weekday() > 4:
        return good_news_index(next_day(date.replace(hour=10, minute=00)))
    # Regular banking day: if published before market opens, considered to be previous day (NASDAQ: 13:30-20:00 GMT)
    if date.hour < 13:
        return previous_day(date).strftime('%Y-%m-%d')
    else:
        return date.strftime('%Y-%m-%d')


def special_preprocess_news(asset, folder, keywords=None, aggregate=False):
    filename = folder + asset.lower() + '_news.csv'
    df = pd.read_csv(filename, encoding='utf-8', index_col=0).sort_index()
    if keywords is not None:
        pattern = '(?i)' + '|'.join(keywords)
        df = df[df['title'].str.contains(pattern) | df['summary'].str.contains(pattern)]
    df = df[df['title'].notnull() & df['summary'].notnull()]
    df['summary'] = df['summary'].apply(clean_string)
    df['datetime'] = pd.to_datetime(df['publication_date'], format='%Y-%m-%dT%H:%M:%S.%fZ')
    # If monday, use weekend news
    df.index = df['datetime'].apply(good_news_index)
    if aggregate:
        news = df[['title', 'summary']].groupby(df.index).apply(sum)
    else:
        news = df[['title', 'summary']]
    return news


def encode_texts(tokenizer, texts, max_length=512):
    token_ids = np.zeros(shape=(len(texts), max_length), dtype=np.int32)
    for i, text in enumerate(texts):
        encoded = tokenizer.encode(text, max_length=max_length, truncation='longest_first')
        token_ids[i, 0:len(encoded)] = encoded
    attention_mask = (token_ids != 0).astype(np.int32)
    return {"input_ids": token_ids, "attention_mask": attention_mask}


def load_texts():
    folder = '../data/intrinio/'
    res = pd.DataFrame()

    for asset in ['AAPL']:  # KEYWORDS.keys():
        file = f'{folder}{asset.lower()}_prices.csv'
        df = pd.read_csv(file, encoding='utf-8', index_col=0)
        df.index = df.index.rename('date')
        df = df.loc[~df.index.duplicated(keep='last')].sort_index()
        df.drop([col for col in df if col not in ['open', 'close']], axis=1, inplace=True)
        df['labels'] = ((df['close'].shift(-2) - df['open'].shift(1)) > 0).astype(int)
        df.drop(['open', 'close'], axis=1, inplace=True)
        news = special_preprocess_news(asset, folder, KEYWORDS[asset], aggregate=False)
        news['labels'] = [df[df.index == n]['labels'].values[0] if len(df[df.index == n]['labels']) > 0 else None
                          for n in news.index]
        news.dropna(inplace=True)
        res = pd.concat([res, news], axis=0)

    res = res.rename_axis('date').sort_values(['date'])
    res = res[res['summary'] != ''].dropna()

    train = balance(res[(res.index >= T0) & (res.index < T1)].sample(frac=1, random_state=123))
    valid = balance(res[(res.index >= T1) & (res.index < T2)].sample(frac=1, random_state=123))
    test = balance(res[(res.index >= T2)].sample(frac=1, random_state=123))

    return train['summary'].to_list(), train['labels'].to_list(), \
           valid['summary'].to_list(), valid['labels'].to_list(), \
           test['summary'].to_list(), test['labels'].to_list()


def test_model():
    _, _, _, _, test_texts, y_test = load_texts()
    X_test = encode_texts(tokenizer, test_texts)

    t1 = datetime.now()
    print('Predicting...')
    y_pred = model.predict(X_test, verbose=1, batch_size=10)
    y_pred = np.argmax(y_pred[0], axis=1)
    print('V2:', datetime.now() - t1)
    print_metrics(y_test, y_pred)


def train_model():
    train_texts, y_train, val_texts, y_val, test_texts, y_test = load_texts()
    X_train = encode_texts(tokenizer, train_texts)
    X_val = encode_texts(tokenizer, val_texts)
    X_test = encode_texts(tokenizer, test_texts)
    y_train, y_val, y_test = np.array(y_train), np.array(y_val), np.array(y_test)
    del train_texts, val_texts, test_texts

    print(f'Training on {len(y_train)} observation, validating on {len(y_val)}, testing on {len(y_test)}.')
    print(f'Balance on each dataset: {y_train.mean()}, {y_val.mean()}, {y_test.mean()}.')

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5, epsilon=1e-08)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    model.summary()

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='auto', patience=1,
                                                verbose=1, restore_best_weights=True)

    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=30, batch_size=10, callbacks=[callback], verbose=1)

    model.save_weights(MODEL_PATH + 'finetuned_weights.h5')

    y_pred = model.predict(X_test, verbose=1, batch_size=90)
    y_pred = np.argmax(y_pred[0], axis=1)
    print_metrics(y_test, y_pred)


def print_metrics(y_true, y_pred):
    print("Accuracy         : {:.2f}".format(100 * metrics.accuracy_score(y_true, y_pred)))
    print("F1-Score (micro) : {:.2f}".format(100 * metrics.f1_score(y_true, y_pred, average='micro')))
    print("F1-Score (macro) : {:.2f}".format(100 * metrics.f1_score(y_true, y_pred, average='macro')))
    print(metrics.classification_report(y_true, y_pred, digits=4, labels=[0, 1]))


def predict_sentiment(texts):
    batch = encode_texts(tokenizer, texts, max_length=512)
    outputs = model.predict(batch, verbose=1, batch_size=20)
    return np.argmax(tf.nn.softmax(outputs[0], axis=-1), axis=1)


def simple_indicators():

    texts = special_preprocess_news(asset='AAPL', folder='../data/intrinio/', aggregate=False)
    texts = texts[texts.index > '2019']
    texts['sentiment'] = predict_sentiment(texts['summary'].to_list())
    count = texts.groupby(texts.index).count()['summary'].rename('count')
    sent = texts.groupby(texts.index).mean()['sentiment'].rename('sentiment')
    prices = pd.read_csv('../data/yahoo/aapl_prices.csv', encoding='utf-8', index_col=0)
    prices['labels'] = ((prices['close'].shift(-5) - prices['open']) > 0).astype(int)
    res = pd.concat([special_preprocess_news(asset, '../data/intrinio/', False) for asset in KEYWORDS.keys()])
    res = res.groupby(res.index).count()['summary'].rename('total')
    df = pd.concat([count, sent, prices[['close', 'labels']], res], axis=1).dropna()
    df['freq'] = 100 * df['count'] / df['total']
    df.to_csv('../data/essai.csv', encoding='utf-8')
    # df = pd.read_csv('../data/essai.csv', index_col=0, encoding='utf-8')
    df['indice'] = (2 * df['sentiment'] - 1) * df['count']
    nice_plot(ind=df.index.to_list(), curves_list=[df['indice'].rolling(10).mean().to_list(), df['close'].to_list()],
              names_list=['Indice of AAPL', 'Close price'], title='')
    print(df.corr())


if __name__ == '__main__':
    train_model()
    # test_model()
    # simple_indicators()
