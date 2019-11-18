#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 14:34:18 2019

@author: rohit
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow
from keras import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from pandas import DataFrame

train = pd.read_csv("train_XY.csv", index_col=False)
test = pd.read_csv('test.csv', index_col=False)
test.rename(columns={"Unnamed: 0": 'id'}, inplace=True)
test['id'] = list(range(1, 10001))
res = DataFrame()
res['id'] = test['id']

valid = train.iloc[0:5000, :]  # 构造验证集
train = train.iloc[5000:, :]  # 构造训练集
train_x = train.drop(['id', 'label'], axis=1)  # 得到训练特征
train_y = train['label']  # 得到训练标签
valid_x = valid.drop(['id', 'label'], axis=1)  # 得到验证集特征
valid_y = valid['label']  # 得到验证集标签
test_x = test.drop('id', axis=1)

test_x = test_x / 255
train_x = train_x / 255
valid_x = valid_x / 255

train_y = train_y.to_numpy()
valid_y = valid_y.to_numpy()

train_feature = train_x.to_numpy().reshape((-1, 28, 28, 1))
test_feature = test_x.to_numpy().reshape((-1, 28, 28, 1))
valid_feature = valid_x.to_numpy().reshape((-1, 28, 28, 1))
print(train_feature.shape, valid_feature.shape, test_feature.shape)
# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Conv2D(16, (3, 3), input_shape=(28, 28, 1), activation='relu'))

classifier.add(Conv2D(32, kernel_size=3, padding="same", activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Dropout(0.25))

classifier.add(Conv2D(64, kernel_size=3, padding="same", activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Dropout(0.25))

classifier.add(Conv2D(128, kernel_size=3, padding="same", activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Dropout(0.25))

classifier.add(Conv2D(256, kernel_size=3, padding="same", activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Dropout(0.25))

classifier.add(Flatten())
classifier.add(Dense(256, activation='relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(10, activation='softmax'))

# Compiling the ANN
classifier.compile(loss='sparse_categorical_crossentropy', optimizer="Adam", metrics=['accuracy'], verbose=2,
                   callbacks=[EarlyStopping(patience=5, restore_best_weights=True)], shuffle=True,
                   validation_data=(valid_feature, valid_y))

# Fitting the ANN to the Training set
classifier.fit(train_feature, train_y, batch_size=128, epochs=1)

# Part 3 - Making predictions and evaluating the classifier

# Predicting the Test set results
y_pred = classifier.predict(test_feature)
results = np.argmax(y_pred, axis=1)

data_out = pd.DataFrame({'id': range(len(test_feature)), 'label': results})
data_out.to_csv('submission.csv', index=None)
