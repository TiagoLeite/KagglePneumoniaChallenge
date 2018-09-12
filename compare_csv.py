import pandas as pd
import os

df = pd.read_csv('results_mobnet_v1_fpn/resultados_mobilenet_0176_c35.csv')
vazio = 0
six = 0
five = 0
four = 0
three, two, one = 0, 0, 0


def isNaN(num):
    return num != num


for count in range(1000):

    line = df['PredictionString'][count]

    if not isNaN(line):

        tokens = line.split(' ')

        for k in range(int(len(tokens)/5)):

            confidence = float(tokens[k*5])

            if confidence >= 0.6:
                six += 1
            elif confidence >= 0.5:
                five += 1
            elif confidence >= 0.4:
                four += 1
            elif confidence >= 0.3:
                three += 1
            elif confidence >= 0.2:
                two += 1
            elif confidence >= 0.1:
                one += 1
            else:
                vazio += 1
    else:
        vazio += 1

print("Sem box detection: ", vazio, "out of 1000")
print("Maior   que   0.6: ", six)
print("Entre   0.5 - 0.6: ", five)
print("Entre   0.4 - 0.5: ", four)
print("Entre   0.3 - 0.4: ", three)
print("Entre   0.2 - 0.3: ", two)
print("Menor   que   0.2: ", one)

# df.to_csv('resultados_rcnn_resnet_50_c35.csv', index = False)

