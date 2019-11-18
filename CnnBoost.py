# Author By Gary1111

import numpy as np
import pandas as pd
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

train = pd.read_csv("data/DataPre/train_XY.csv", index_col=False)
test = pd.read_csv('data/DataPre/test.csv', index_col=False)
test.rename(columns={"Unnamed: 0": 'id'}, inplace=True)
test['id'] = list(range(1, 10001))
train_x = train.drop(['id', 'label'], axis=1)  # 得到训练特征
train_y = train['label']  # 得到训练标签
test_x = test.drop('id', axis=1)

test_x = test_x / 255
train_x = train_x / 255
train_y = train_y.to_numpy()
train_feature = train_x.to_numpy().reshape((-1, 28, 28, 1))
test_feature = test_x.to_numpy().reshape((-1, 28, 28, 1))
print(train_feature.shape, test_feature.shape)

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
classifier.compile(loss='sparse_categorical_crossentropy', optimizer="Adam", metrics=['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(train_feature, train_y, batch_size=1, epochs=100)

# Part 3 - Making predictions and evaluating the classifier

# Predicting the Test set results
y_pred = classifier.predict(test_feature)
results = np.argmax(y_pred, axis=1)

data_out = pd.DataFrame({'id': range(1, 10001), 'label': results})
data_out.to_csv('submission.csv', index=None)
