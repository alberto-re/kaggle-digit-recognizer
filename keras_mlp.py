#!/usr/bin/env python3
"""
Digit recognizer based on CNN

see: https://www.kaggle.com/c/digit-recognizer
"""

import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

SEED = 56843

BATCH_SIZE = 100
NUM_CLASSES = 10
EPOCHS = 10

IMG_ROWS, IMG_COLS = 28, 28

SAVE_PATH = "./data/model.h5"


def train_model():
    """
    Train and save a net
    """
    train = pd.read_csv("./data/train.csv", encoding="utf-8")

    (train, test) = train_test_split(train, random_state=SEED, test_size=0.3)

    x_train = train.drop(["label"], axis=1).values
    y_train = train["label"].values
    x_test = test.drop(["label"], axis=1).values
    y_test = test["label"].values

    x_train = x_train.reshape(x_train.shape[0], IMG_ROWS, IMG_COLS, 1)
    x_test = x_test.reshape(x_test.shape[0], IMG_ROWS, IMG_COLS, 1)
    input_shape = (IMG_ROWS, IMG_COLS, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    model.fit(
        x_train,
        y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        verbose=1,
        validation_data=(x_test, y_test))

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Accuracy:', score[1])

    model.save(SAVE_PATH)


def predict():
    """
    Predict labels using pre-trained net
    """
    validation = pd.read_csv("./data/test.csv", encoding="utf-8")

    x_validation = validation.values

    x_validation = x_validation.reshape(
        x_validation.shape[0], IMG_ROWS, IMG_COLS, 1)

    x_validation = x_validation.astype('float32')

    x_validation /= 255

    model = load_model(SAVE_PATH)
    pred = model.predict(x_validation)

    validation["Label"] = 1
    validation["Label"] = np.argmax(pred, axis=1)
    validation.index = np.arange(1, len(validation) + 1)
    validation.index.name = "ImageId"
    validation.to_csv('./data/mlp.csv', columns=["Label"], encoding="utf-8")


def main(argv):
    """
    Application logic entry point
    """

    if len(argv) < 1:
        print("Missing command. Valid commands are: train, predict", file=sys.stderr)
        sys.exit(1)

    if argv[0] == "train":
        train_model()
    elif argv[0] == "predict":
        predict()
    else:
        print("Unknown command %s" % argv[0], file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main(sys.argv[1:])
