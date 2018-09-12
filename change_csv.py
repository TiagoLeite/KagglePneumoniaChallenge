import pandas as pd


data = pd.read_csv('resultados_mobilenet_test.csv')
THRESHOLD = 0.35  # ----->>>>>> change this according to your preference

new_strings = list()

for line in data['PredictionString']:
    if type(line) is str:
        tokens = line.split(' ')
        new_line = ""
        for k in range(int(len(tokens)/5)):
            if float(tokens[k * 5]) >= THRESHOLD:
                for i in range(k * 5, k * 5 + 5):
                    new_line += tokens[i] + " "
        new_strings.append(new_line)
    else:
        new_strings.append("")


new_file = pd.DataFrame({'patientId': data['patientId'], 'PredictionString': new_strings})

new_file.to_csv('resultados_mobilenet_test_out_2.csv', index=False)
