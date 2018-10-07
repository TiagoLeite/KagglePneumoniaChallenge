import pandas as pd
import numpy as np


def max(a, b):
    if a >= b:
        return a
    return b


def min(a, b):
    if a < b:
        return a
    return b


def calc_intersect_opt(rect1, rect2):
    x1 = int(round(float(rect1[1]), 0))
    y1 = int(round(float(rect1[2]), 0))
    w1 = int(round(float(rect1[3]), 0))
    h1 = int(round(float(rect1[4]), 0))

    w1 = x1 + w1
    h1 = y1 + h1

    # print(x1, y1, w1, h1)

    x2 = int(round(float(rect2[1]), 0))
    y2 = int(round(float(rect2[2]), 0))
    w2 = int(round(float(rect2[3]), 0))
    h2 = int(round(float(rect2[4]), 0))

    w2 = x2 + w2
    h2 = y2 + h2

    new_x = max(x1, x2)
    new_y = max(y1, y2)

    new_w = min(w1, w2)
    new_h = min(h1, h2)

    if ((new_w > new_x) and (new_h > new_y)):
        return rect1[0], new_x, new_y, new_w - new_x, new_h - new_y

    return rect1[0], 0, 0, 0, 0


def calc_intersec(rect1, rect2, size=1024):
    mat1 = np.zeros(shape=[size, size])
    mat2 = np.zeros(shape=[size, size])

    x1 = int(round(float(rect1[1]), 0))
    y1 = int(round(float(rect1[2]), 0))
    w1 = int(round(float(rect1[3]), 0))
    h1 = int(round(float(rect1[4]), 0))

    # print(x1, y1, w1, h1)

    x2 = int(round(float(rect2[1]), 0))
    y2 = int(round(float(rect2[2]), 0))
    w2 = int(round(float(rect2[3]), 0))
    h2 = int(round(float(rect2[4]), 0))

    for i in range(x1, (x1 + w1 + 1)):
        for j in range(y1, (y1 + h1 + 1)):
            mat1[i][j] = 1

    for i in range(x2, (x2 + w2 + 1)):
        for j in range(y2, (y2 + h2 + 1)):
            mat2[i][j] = 1

    mat = mat1 + mat2

    for k in range(len(mat[0])):
        for m in range(len(mat[:][0])):
            mat[k][m] = 1 if mat[k][m] == 2 else 0

    x, y, w, h = 0, 0, 0, 0
    ok = False
    for k in range(len(mat[0])):
        if ok:
            break
        for m in range(len(mat[:][0])):
            if mat[k][m] == 1:
                x = k
                y = m
                ok = True
                break
    ok = False
    for k in reversed(range(len(mat[0]))):
        if ok:
            break
        for m in reversed(range(len(mat[:][0]))):
            if mat[k][m] == 1:
                w = k
                h = m
                ok = True
                break

    return rect1[0], x, y, w - x, h - y


# t = calc_intersec([50, 50, 300, 300], [100, 100, 251, 251])

data = pd.read_csv('scored_194.csv')
# data = data.sort_values(by=['patientId'])
ids = data['patientId']
new_strings = list()

for line in data['PredictionString']:
    if type(line) is str:
        tokens = line.split(' ')
        rects = list()
        for k in range(int(len(tokens) / 5)):
            rects.append(tokens[k * 5: k * 5 + 5])
        # print(rects)
        new_strings.append(rects)
    else:
        new_strings.append("")

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
new_file.to_csv('scored_194.csv', index=False)

# COMO USAR:
'''
-Rode 3 vezes
    1 vez: assim mesmo

    2 vez: troque linhas 99-108 por:

    if size >= 3:
        inter = calc_intersec(rect[1], rect[2])
        print(new_strings[k], inter)
        if inter != (rect[1][0], 0, 0, 0, 0):
            cont += 1
            print(cont)
            rect.pop(1)
            rect.pop(1)
            new_strings[k].append(list(inter))
        print(new_strings[k], '\n')

    3 vez: troque linhas 82-91 por:

     if size >= 2:
        inter = calc_intersec(rect[0], rect[1])
        print(new_strings[k], inter)
        if inter != (rect[0][0], 0, 0, 0, 0):
            cont += 1
            print(cont)
            rect.pop(0)
            rect.pop(0)
            new_strings[k].append(list(inter))
        print(new_strings[k], '\n')
 --- Mantenha os mesmos nomes dos arquivos de entrada e saida (results.csv ou outro nome que preferir)
'''
