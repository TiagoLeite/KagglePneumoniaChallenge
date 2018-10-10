import pandas as pd
import os
import shutil

images = pd.read_csv('data/no_lung_opacity.csv')


ids = images['patientId']

cont = 0

for id in ids:
    shutil.copy('data/images/train_jpg/' + id + '.jpg', 'data/images/no_lung_opacity_images/')
    print(id, cont, 'ok')not
    cont += 1

