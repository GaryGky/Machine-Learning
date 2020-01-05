import pandas as pd

from RandomForest.Forest import RandomForestClassifier

if __name__ == '__main__':
    train = pd.read_csv("train.csv", index_col=None)
    train_y = train[['label', 'index']]
    train.drop(['label', 'index'], axis=1)

    test = pd.read_csv('test.csv', index_col=False)
    test_y = test['label']
    test.drop(['label', 'index'], axis=1)

    clf = RandomForestClassifier(n_estimators=10,
                                 max_depth=10,
                                 min_samples_split=10,
                                 min_samples_leaf=9,
                                 colsample_bytree="sqrt",
                                 subsample=0.8,
                                 random_state=17373290)
    clf.fit(train, train_y)
    label = test_y.to_numpy()
    predict_label = clf.predict(test)
    print('real label is: ', label)
    print('predict label is: ', predict_label)  # 数组类型
    Tp = 0
    Fp = 0
    Tn = 0
    Fn = 0
    for i in range(0, test_y.shape[0]):
        # print(index)
        # print(row)
        if (label[i] == 1 and predict_label[i] == 1):
            Tp += 1
        elif label[i] == -1 and predict_label[i] == 1:
            Fp += 1
        elif label[i] == -1 and predict_label[i] == -1:
            Tn += 1
        elif label[i] == 1 and predict_label[i] == -1:
            Fn += 1
    Pre = Tp / (Tp + Fp)
    Rec = Tp / (Tp + Fn)
    F1 = 2 * Pre * Rec / (Pre + Rec)
    print("TP: ", Tp,
          "Tn: ", Tn,
          "Fn: ", Fn,
          "Fp: ", Fp)
    print("F1 score: ", F1)
