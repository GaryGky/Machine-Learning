import tensorflow as tf
import pandas as pd
import numpy as np
import random

from keras import Sequential
from keras.layers import Conv2D, Dropout, MaxPool2D, Flatten, Dense, Convolution2D, LeakyReLU, BatchNormalization

model = Sequential()

model.add(Convolution2D(32, (3, 3), padding='same', use_bias=False, input_shape=(96, 96, 1)))
model.add(LeakyReLU(alpha=0.1))
model.add(BatchNormalization())

model.add(Convolution2D(32, (3, 3), padding='same', use_bias=False))
model.add(LeakyReLU(alpha=0.1))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Convolution2D(64, (3, 3), padding='same', use_bias=False))
model.add(LeakyReLU(alpha=0.1))
model.add(BatchNormalization())

model.add(Convolution2D(64, (3, 3), padding='same', use_bias=False))
model.add(LeakyReLU(alpha=0.1))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Convolution2D(96, (3, 3), padding='same', use_bias=False))
model.add(LeakyReLU(alpha=0.1))
model.add(BatchNormalization())

model.add(Convolution2D(96, (3, 3), padding='same', use_bias=False))
model.add(LeakyReLU(alpha=0.1))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Convolution2D(128, (3, 3), padding='same', use_bias=False))
model.add(LeakyReLU(alpha=0.1))
model.add(BatchNormalization())

model.add(Convolution2D(128, (3, 3), padding='same', use_bias=False))
model.add(LeakyReLU(alpha=0.1))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Convolution2D(256, (3, 3), padding='same', use_bias=False))
model.add(LeakyReLU(alpha=0.1))
model.add(BatchNormalization())

model.add(Convolution2D(256, (3, 3), padding='same', use_bias=False))
model.add(LeakyReLU(alpha=0.1))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Convolution2D(512, (3, 3), padding='same', use_bias=False))
model.add(LeakyReLU(alpha=0.1))
model.add(BatchNormalization())

model.add(Convolution2D(512, (3, 3), padding='same', use_bias=False))
model.add(LeakyReLU(alpha=0.1))
model.add(BatchNormalization())

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(196))
model.summary()

TESTING = "test.csv"
TRAINING = "train.csv"


def input_data(test=False):
    file_name = TESTING if test else TRAINING
    df = pd.read_csv(file_name)
    cols = df.columns[:-1]  # 标签列

    # dropna()是丢弃有缺失数据的样本，这样最后7000多个样本只剩2140个可用的。
    df = df.dropna()
    df['Image'] = df['Image'].apply(lambda img: np.fromstring(img, sep=' ') / 255.0)

    X = np.vstack(df['Image'])
    X = X.reshape((-1, 96, 96, 1))

    if test:
        y = None
    else:
        y = df[cols].values / 96.0  # 将y值缩放到[0,1]区间

    return X, y


X_train, y_train = input_data()
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
model.fit(X_train, y_train, epochs=50, batch_size=256, validation_split=0.2, shuffle=True)
# 加载测试集

X_test, y_test = input_data(test=True)  # 加载测试集
pred = model.predict(X_test)
print(pred)
