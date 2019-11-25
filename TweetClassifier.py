import sys
import os

import numpy as np
import string
import pandas as pd
from io import StringIO
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dense, LSTM, Dropout, Bidirectional, Embedding
from nltk.tokenize import TweetTokenizer


def import_data(path, delimiter="|"):
    """
    Prepares data by
    :param path:
    :return:
    """
    f = open(path, "r").readlines()
    f = [line.encode('ascii', 'ignore').decode('ascii').lower() for line in f]  # strip emoji from text, make lowercase
    f = "".join(f)  # turn into singular string
    df = pd.read_csv(StringIO(f), sep=delimiter, error_bad_lines=False)  # Currently skipping bad lines, should prune later
    replacement_dict = {"Twitter for iPhone".lower() : 1, "Twitter for Android".lower() : 0}
    df = df.replace(to_replace=replacement_dict)   # replace labels

    df = df[df.source.apply(lambda x: type(x) == int)]  # remove tweets from other sources

    return df.sample(frac=1) # shuffle and return

def split_data(df, split=0.8):
    training = df.sample(frac=split, random_state=0)
    testing = df.drop(training.index)

    return training, testing

def tokenize(data):
    tokenizer = TweetTokenizer()
    data['tokenized_text'] = data['text'].apply(tokenizer.tokenize)

    return data, tokenizer

def build_classifier():
    model = Sequential()
    model.add(Embedding(len(words), 32))
    model.add(Bidirectional(LSTM(64)))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    return model

def train_model(model, training_data):
    x_train = training_data["text"]
    y_train = training_data["source"]

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=30,
                        verbose=1,
                        validation_split=0.1)

if __name__ == "__main__":
    data = import_data("/home/gringle/Downloads/trump-tweets.csv")
    data, tokenizer = tokenize(data)
    training, testing = split_data(data)

    training_tweets = training["tokenized_text"]
    training_labels = training["source"]
    testing_tweets = testing["tokenized_text"]
    testing_labels = testing["source"]

    print(data)
    exit(5)
    model = build_classifier()