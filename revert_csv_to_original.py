import pandas as pd

data = pd.read_csv('results_mobilenet_v1_no_lung_opacity_intersected.csv')
# data = data.sort_values(by=['patientId'])
ids = list()
x = list()
y = list()
width = list()
height = list()
target = list()

for index, line in data.iterrows():

    print(line['patientId'], line['PredictionString'])
    id = line['patientId']

    # print(str(line['PredictionString']))

    if type(line['PredictionString']) is str:
        tokens = str(line['PredictionString']).split(' ')
        for k in range(int(len(tokens) / 5)):
            ids.append(id)
            x.append(tokens[k * 5 + 1])
            y.append(tokens[k * 5 + 2])
            width.append(tokens[k * 5 + 3])
            height.append(tokens[k * 5 + 4])
            target.append('not_normal')
            # new_strings.append(tokens[k * 5 + 1: k * 5 + 5])
    # else:
    #    ids.append(id)
    #    x.append('')
    #    y.append('')
    #    width.append('')
    #    height.append('')
    #    target.append('not_normal')

# print(ids)
# print(new_strings)

new_file = pd.DataFrame({'patientId': ids, 'x': x, 'y': y, 'width': width, 'height': height, 'Target': target})
new_file.to_csv('train_not_normal.csv', index=False)

'''
cont = 0
for k in range(len(new_strings)):
    rect = new_strings[k]
    size = len(new_strings[k])
    if size >= 2:
        for j in range(0, size-1):
            if j >= len(new_strings[k])-1:
                break
            print('CALC\n\n')
            inter = calc_intersect_opt(new_strings[k][j], new_strings[k][j+1])
            print(new_strings[k], inter)
            if inter != (new_strings[k][j][0], 0, 0, 0, 0):
                cont += 1
                print(cont)
                new_strings[k].pop(j)
                new_strings[k].pop(j)
                new_strings[k].append(list(inter))
                # j = j - 1
            print(new_strings[k], '\n')

print(cont)

for k in range(len(new_strings)):
    stri = ""
    for j in range(len(new_strings[k])):
        for i in range(len(new_strings[k][j])):
            stri += str(new_strings[k][j][i]) + " "
    new_strings[k] = stri

new_file = pd.DataFrame({'patientId': ids, 'PredictionString': new_strings})
new_file.to_csv('results_mobilenet_v1_no_lung_opacity_intersected.csv', index=False)
'''
