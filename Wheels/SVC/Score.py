import pandas as pd

predict = pd.read_csv('predict.csv', index_col=None)
standard = pd.read_csv('test.csv', index_col=None)
standard = standard[['index', 'label']]

print(predict.shape)
print(standard.shape)

final = pd.concat([standard, predict], axis=1)
final = final.T.drop_duplicates().T  # 去掉重复的列
Tp = 0
Fp = 0
Tn = 0
Fn = 0
for index, row in final.iterrows():
    # print(index)
    # print(row)
    if (row['label'] == 1 and row['predict'] == 1):
        Tp += 1
    elif row['label'] == -1 and row['predict'] == 1:
        Fp += 1
    elif row['label'] == -1 and row['predict'] == -1:
        Tn += 1
    elif row['label'] == 1 and row['predict'] == -1:
        Fn += 1
Pre = Tp / (Tp + Fp)
Rec = Tp / (Tp + Fn)
F1 = 2 * Pre * Rec / (Pre + Rec)
print(Tp, Tn, Fn, Fp)
print(F1)
