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

PATH_TO_TEST_IMAGES_DIR = '/home/nathan/Documents/Kaggle_Competition/images/test_jpg_2'
TEST_IMAGE_PATHS = glob.glob(PATH_TO_TEST_IMAGES_DIR+'/*')
counter = 0

for image_path in TEST_IMAGE_PATHS:
    # line = (image_path.split('/')[7].split('.')[0])
    line = (image_path.split('/')[7])
    img_original = cv.imread(image_path)
    TEST_IMAGE_PATHS_NEW = "/home/nathan/Documents/Kaggle_Competition/images/test_jpg_2_correction/"+line
    res = cv.convertScaleAbs(img_original, alpha=alpha, beta=beta)
    cv.imwrite(TEST_IMAGE_PATHS_NEW,res)
    if counter%100==0:
        print(counter)
    counter +=1
