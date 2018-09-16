import pandas as pd

data = pd.read_csv('data/stage_1_train_labels.csv')

data_res = pd.read_csv('results_4.csv')

data_filtered = data.loc[data['patientId'].isin(data_res['patientId'])]

data_filtered = data_filtered.sort_values(by=['patientId'])

ids = list()
string_pred = list()

for column in data_filtered['patientId'].unique():

    ids.append(column)

    # print(column)

    cur_data = data_filtered[data_filtered['patientId'] == column]

    strg = ""

    if cur_data.iloc[0]['Target'] == 0:
        string_pred.append(strg)

    else:
        for k in range(len(cur_data)):

            strg += cur_data.iloc[k]['x'].astype(str) + " " + cur_data.iloc[k]['y'].astype(str) + " " + cur_data.iloc[k]['width'].astype(str)\
                    + " " + cur_data.iloc[k]['height'].astype(str) + " "

        string_pred.append(strg)

final_file = pd.DataFrame({'patientId': ids, 'PredictionString': string_pred})
final_file.to_csv('results_true.csv', index=False)
