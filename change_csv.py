import pandas as pd


data = pd.read_csv('scored_196.csv')
# THRESHOLD = 0.4  # ----->>>>>> change this according to your preference

data = data.sort_values(by=['patientId'])

# data.to_csv("scored_195_sort.csv", index=False)

df_pneu = pd.read_csv('has_pneu.csv')

df_pneu = df_pneu.sort_values(by=['patientId'])

new_strings = list()

l = 0

for line in data['PredictionString']:
    # print(df_pneu['patientId'][0])
    if type(line) is str:
        tokens = line.split(' ')
        new_line = ""
        for k in range(int(len(tokens)/5)):
            # if float(tokens[k * 5]) >= THRESHOLD:
            x, y, w, h = float(tokens[k*5+1]), float(tokens[k*5+2]), float(tokens[k*5+3]), float(tokens[k*5+4])
            if (df_pneu['has_pneu'][l] == 1) or (w >= 200 and h >= 200):
                for i in range(k * 5, k * 5 + 5):
                    new_line += tokens[i] + " "
        new_strings.append(new_line)
    else:
        new_strings.append("")

    l += 1

new_file = pd.DataFrame({'patientId': data['patientId'], 'PredictionString': new_strings})
new_file.to_csv('scored_180_no_small_boxes_conf.csv', index=False)
