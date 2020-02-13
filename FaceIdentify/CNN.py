import tensorflow as tf
import pandas as pd
import numpy as np
import random

from keras import Sequential
from keras.layers import Conv2D, Dropout, MaxPool2D, Flatten, Dense


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


x = tf.placeholder("float", shape=[None, 96, 96, 1])
y_ = tf.placeholder("float", shape=[None, 196])
keep_prob = tf.placeholder("float")


def model():
    W_conv1 = weight_variable([3, 3, 1, 32])
    b_conv1 = bias_variable([32])

    h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([2, 2, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_conv3 = weight_variable([2, 2, 64, 128])
    b_conv3 = bias_variable([128])

    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
    h_pool3 = max_pool_2x2(h_conv3)

    W_fc1 = weight_variable([11 * 11 * 128, 1024])
    b_fc1 = bias_variable([1024])

    h_pool3_flat = tf.reshape(h_pool3, [-1, 11 * 11 * 128])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

    W_fc2 = weight_variable([1024, 1024])
    b_fc2 = bias_variable([1024])

    h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
    h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

    W_fc3 = weight_variable([1024, 196])
    b_fc3 = bias_variable([196])

    y_conv = tf.matmul(h_fc2_drop, W_fc3) + b_fc3
    rmse = tf.sqrt(tf.reduce_mean(tf.square(y_ - y_conv)))
    return y_conv, rmse


def input_data(test=False):
    file_name = TEST_FILE if test else TRAIN_FILE
    df = pd.read_csv(file_name)
    cols = df.columns[:-1] # 标签列

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


TRAIN_FILE = '../data/ML_training.csv'
TEST_FILE = '../data/ML_testing.csv'
SAVE_PATH = 'model'

VALIDATION_SIZE = 100  # 验证集大小
EPOCHS = 50  # 迭代次数
BATCH_SIZE = 64  # 每个batch大小，稍微大一点的batch会更稳定
EARLY_STOP_PATIENCE = 100  # 控制early stopping的参数

def save_model(saver, sess, save_path):
    path = saver.save(sess, save_path)
    print('model save in :{0}'.format(path))


def get_keypoint_index(): # 关键点索引
    keypoint_index = {}
    for i in range(98):
        keypoint_index['x{}'.format(i)] = 2 * i
        keypoint_index['y{}'.format(i)] = 2 * i + 1
    return keypoint_index

if __name__ == '__main__':
    keypoint_index = get_keypoint_index()

    sess = tf.InteractiveSession()
    y_conv, rmse = model()
    train_step = tf.train.AdamOptimizer(1e-3).minimize(rmse)


    # 变量都要初始化
    sess.run(tf.global_variables_initializer())
    X, y = input_data()
    X_valid, y_valid = X[:VALIDATION_SIZE], y[:VALIDATION_SIZE]
    X_train, y_train = X[VALIDATION_SIZE:], y[VALIDATION_SIZE:]

    best_validation_loss = 1000000.0
    current_epoch = 0
    TRAIN_SIZE = X_train.shape[0]
    train_index = list(range(TRAIN_SIZE))
    random.shuffle(train_index)
    X_train, y_train = X_train[train_index], y_train[train_index]

    saver = tf.train.Saver()

    print('begin training..., train dataset size:{0}'.format(TRAIN_SIZE))
    for i in range(EPOCHS):
        random.shuffle(train_index)  # 每个epoch都shuffle一下效果更好
        X_train, y_train = X_train[train_index], y_train[train_index]

        for j in range(0, TRAIN_SIZE, BATCH_SIZE):
            print('epoch {0}, train {1} samples done...'.format(i, j))

            train_step.run(feed_dict={x: X_train[j:j + BATCH_SIZE],
                                      y_: y_train[j:j + BATCH_SIZE], keep_prob: 0.5})

        # 电脑太渣，用所有训练样本计算train_loss居然死机，只好注释了。
        # train_loss = rmse.eval(feed_dict={x:X_train, y_:y_train, keep_prob: 1.0})
        validation_loss = rmse.eval(feed_dict={x: X_valid, y_: y_valid, keep_prob: 1.0})

        print('epoch {0} done! validation loss:{1}'.format(i, validation_loss * 96.0))
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            current_epoch = i
            save_model(saver, sess, SAVE_PATH)  # 即时保存最好的结果
        elif (i - current_epoch) >= EARLY_STOP_PATIENCE:
            print('early stopping')
            break

    X, y = input_data(test=True)
    y_pred = []

    TEST_SIZE = X.shape[0]
    for j in range(0, TEST_SIZE, BATCH_SIZE):
        y_batch = y_conv.eval(feed_dict={x: X[j:j + BATCH_SIZE], keep_prob: 1.0})
        y_pred.extend(y_batch)

    print('predict test image done!')

    for i in range(1000):
        y_pred[i] = y_pred[i] * 96
    output = pd.DataFrame(y_pred, columns=keypoint_index.keys())
    output.to_csv('../data/ML_submit3.csv')

    '''output_file = open('ML_submit.csv', 'w')
    output_file.write('RowId,Location\n')

    IdLookupTable = open('../IdLookupTable.csv')
    IdLookupTable.readline()

    for line in IdLookupTable:
        RowId, ImageId, FeatureName = line.rstrip().split(',')
        image_index = int(ImageId) - 1
        feature_index = keypoint_index[FeatureName]
        feature_location = y_pred[image_index][feature_index] * 96
        output_file.write('{0},{1}\n'.format(RowId, feature_location))

    output_file.close()
    #IdLookupTable.close()'''
