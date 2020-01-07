import pandas as pd


def concat_file(profile, behavior, process):
    data = pd.merge(profile, behavior, on='用户标识', how='left')
    data = pd.merge(data, process, on='用户标识', how='left')
    data = data.fillna(0)
    return data


profile_test = pd.read_csv('./profile_test.csv')
behavior_test = pd.read_csv('./behavior_test.csv')
process_test = pd.read_csv('./onlyTme.csv', index_col=0)
print(process_test.head())

final_test = concat_file(profile_test, behavior_test, process_test)
final_test.to_csv('model_data/final_test_1.csv', index=False)
