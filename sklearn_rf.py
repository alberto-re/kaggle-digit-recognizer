#!/usr/bin/env python3
"""
Digit recognizer based on Random Forest Classifier

see: https://www.kaggle.com/c/digit-recognizer
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

SEED = 56843


def main():
    """
    Application logic entry point
    """
    train = pd.read_csv("./data/train.csv", header=0, sep=",", encoding="utf-8")
    validation = pd.read_csv("./data/test.csv", header=0, sep=",", encoding="utf-8")

    (train, test) = train_test_split(train, random_state=SEED, test_size=0.3)

    model = RandomForestClassifier(
        random_state=SEED
    )

    model.fit(
        train.drop(["label"], axis=1).values,
        train["label"].values
    )

    score = accuracy_score(
        test["label"].values,
        model.predict(test.drop(["label"], axis=1).values)
    )
    print("Accuracy: %0.4f" % score)

    validation["Label"] = model.predict(validation.values)
    validation.index = np.arange(1, len(validation) + 1)
    validation.index.name = "ImageId"
    validation.to_csv('./data/rf.csv', columns=["Label"], encoding="utf-8")


if __name__ == "__main__":
    main()
