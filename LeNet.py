import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
from pandas import DataFrame

np.random.seed(7)
pd.set_option("max_rows", None)
pd.set_option("max_columns", None)

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

print("start CNN ... ")
LeNet = Sequential([
    Conv2D(
        filters=6,
        kernel_size=(5, 5),
        data_format="channels_last",
        activation="relu"
    ),
    MaxPool2D(pool_size=(2, 2), strides=2, data_format="channels_last"),
    Conv2D(
        filters=16,
        kernel_size=(5, 5),
        data_format="channels_last",
        activation="relu"
    ),
    MaxPool2D(pool_size=(2, 2), strides=2, data_format="channels_last"),
    Flatten(),
    Dense(units=120, activation="sigmoid"),
    Dropout(0.5),
    Dense(units=84, activation="sigmoid"),
    Dropout(0.5),
    Dense(units=10, activation="softmax")
])
# 优化器adam
# 损失函数 :: 交叉熵
LeNet.compile(optimizer="sgd", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# 先把迭代次数调成1 :: 看看能否正常输出
# batch_size 也设置小点
LeNet.fit(x=train_feature, y=train_y, batch_size=256, epochs=100, verbose=2,
          callbacks=[EarlyStopping(patience=5, restore_best_weights=True)], shuffle=True,
          validation_data=(valid_feature, valid_y))
res['label'] = np.argmax(LeNet.predict(test_feature), axis=1)
print(res.head())
res.to_csv('submission.csv', index=False)
