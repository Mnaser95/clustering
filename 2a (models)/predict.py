#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Karel Roots"

import numpy as np

"""
@input - model (Object); testing data (List); testing labels (List); model name (String); optional args
         
Method that predicts target values with given model and calculates the accuracy of the predicitions by mean value of correct answers.

@output - Accuracy value (float), truth values (list)
"""


def predict_accuracy(model, X_test, y_test, model_name, multi_branch):
    if multi_branch:
        probs = model.predict([X_test, X_test, X_test])
    else:
        probs = model.predict(X_test)

    preds = probs.argmax(axis=-1)
    equals = preds == y_test.argmax(axis=-1)
    acc = np.mean(equals)

    print("Classification accuracy for %s : %f " % (model_name, acc))

    return acc, equals, preds
