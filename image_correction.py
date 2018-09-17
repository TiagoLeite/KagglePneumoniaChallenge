from __future__ import print_function
from __future__ import division
import cv2 as cv
import numpy as np
import argparse

from matplotlib import pyplot as plt
import glob, pylab, pandas as pd
import os

alpha = 1.0
alpha_max = 500
beta = 0
beta_max = 200
gamma = 1.0
gamma_max = 200

PATH_TO_TEST_IMAGES_DIR = '/home/nathan/Documents/Kaggle_Competition/images/train_jpg_2'

df = pd.read_csv('train_labels.csv')
# Choose image resizer btw 0-1
resize_factor = 0.5
number_images = 2

def basicLinearTransform():
    res = cv.convertScaleAbs(img_original, alpha=alpha, beta=beta)
    img_corrected = cv.hconcat([img_original, res])
    cv.imshow("Brightness and contrast adjustments", img_corrected)

def gammaCorrection():
    ## [changing-contrast-brightness-gamma-correction]
    lookUpTable = np.empty((1,256), np.uint8)
    for i in range(256):
        lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)

    res = cv.LUT(img_original, lookUpTable)
    ## [changing-contrast-brightness-gamma-correction]

    img_gamma_corrected = cv.hconcat([img_original, res]);
    cv.imshow("Gamma correction", img_gamma_corrected);

def on_linear_transform_alpha_trackbar(val):
    global alpha
    alpha = val / 100
    basicLinearTransform()

def on_linear_transform_beta_trackbar(val):
    global beta
    beta = val - 100
    basicLinearTransform()

def on_gamma_correction_trackbar(val):
    global gamma
    gamma = val / 100
    gammaCorrection()

while not (cv.waitKey(25) & 0xFF == ord('q')):
    for count in range(number_images):
        patientId = df['patientId'][count]
        TEST_IMAGE_PATHS = glob.glob(PATH_TO_TEST_IMAGES_DIR+'/'+patientId+'.jpg')
        img = cv.imread(TEST_IMAGE_PATHS[0],1)
        res = cv.resize(img,None,fx=resize_factor, fy=resize_factor, interpolation = cv.INTER_CUBIC)

        x = float(df['x'][count]*resize_factor)
        y = float(df['y'][count]*resize_factor)
        width = float(df['width'][count]*resize_factor)
        height = float(df['height'][count]*resize_factor)

        pts_box = np.array([[x,y],[x,y+height],[x+width,y+height],[x+width,y]], np.int32)
        cv.polylines(res, [pts_box], True, (0,255,0), 3)

        img_original = res
        if img_original is None:
            print('Could not open or find the image: ', TEST_IMAGE_PATHS[0])
            exit(0)

        img_corrected = np.empty((img_original.shape[0], img_original.shape[1]*2, img_original.shape[2]), img_original.dtype)
        img_gamma_corrected = np.empty((img_original.shape[0], img_original.shape[1]*2, img_original.shape[2]), img_original.dtype)

        img_corrected = cv.hconcat([img_original, img_original])
        img_gamma_corrected = cv.hconcat([img_original, img_original])

        cv.namedWindow('Brightness and contrast adjustments')
        alpha_init = int(alpha *100)
        cv.createTrackbar('Alpha gain (contrast)', 'Brightness and contrast adjustments', alpha_init, alpha_max, on_linear_transform_alpha_trackbar)
        beta_init = beta + 100
        cv.createTrackbar('Beta bias (brightness)', 'Brightness and contrast adjustments', beta_init, beta_max, on_linear_transform_beta_trackbar)
        on_linear_transform_alpha_trackbar(alpha_init)

        cv.waitKey(0)

        cv.namedWindow('Gamma correction')
        gamma_init = int(gamma * 100)
        cv.createTrackbar('Gamma correction', 'Gamma correction', gamma_init, gamma_max, on_gamma_correction_trackbar)
        on_gamma_correction_trackbar(gamma_init)

        cv.waitKey(0)
        cv.destroyAllWindows()

    cv.destroyAllWindows()
    break
