import pandas as pd

data = pd.read_csv('data/train_labels_yes_no.csv')

data_res = pd.read_csv('results.csv')

data_filtered = data.loc[data['patientId'].isin(data_res['patientId'])]
# file = pd.DataFrame({'patientId': ids, 'PredictionString': strings})

data_filtered = data_filtered.sort_values(by=['patientId'])
data_filtered.to_csv('results_true.csv', index=False)
