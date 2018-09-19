from __future__ import print_function
from __future__ import division
import cv2 as cv
import numpy as np
import argparse

from matplotlib import pyplot as plt
import glob, pylab, pandas as pd
import os

alpha = 1.2
beta = -20

PATH_TO_TEST_IMAGES_DIR = '/home/nathan/Documents/Kaggle_Competition/images/train_jpg_2'

df = pd.read_csv('/home/nathan/Documents/Kaggle_Competition/data/train_labels.csv')
resize_factor = 1
number_images = len(df)
# number_images = 2
counter = 0

for count in range(number_images):
    patientId = df['patientId'][count]
    if count!=0 and (df['patientId'][count]==df['patientId'][count-1]):
        None
    else:
        TEST_IMAGE_PATHS = glob.glob(PATH_TO_TEST_IMAGES_DIR+'/'+patientId+'.jpg')
        img_original = cv.imread(TEST_IMAGE_PATHS[0],1)
        TEST_IMAGE_PATHS_NEW = "/home/nathan/Documents/Kaggle_Competition/images/train_jpg_2_correction/"+str(patientId)+".jpg"
        res = cv.convertScaleAbs(img_original, alpha=alpha, beta=beta)
        cv.imwrite(TEST_IMAGE_PATHS_NEW,res)
        if counter%100==0:
            print(counter)
        counter +=1
