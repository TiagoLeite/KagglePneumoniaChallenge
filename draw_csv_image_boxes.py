import cv2 as cv
import numpy as np
import argparse
from matplotlib import pyplot as plt
import glob, pylab, pandas as pd
import os

def isNaN(num):
    return num != num
# Path to test images
PATH_TO_TEST_IMAGES_DIR = '/home/nathan/Documents/Kaggle_Competition/images/test_jpg_2'
# Best result 0.174
df = pd.read_csv('/home/nathan/Documents/Kaggle_Competition/results_mobnet_v1_fpn/resultados_mobilenet_0176_c35.csv')
# Choose image resizer btw 0-1
resize_factor = 0.5
number_images = 200
num_images = 0

for count in range(number_images):
    patientId = df['patientId'][count]
    TEST_IMAGE_PATHS = glob.glob(PATH_TO_TEST_IMAGES_DIR+'/'+patientId+'.jpg')
    if not isNaN(df['PredictionString'][count]):
        img = cv.imread(TEST_IMAGE_PATHS[0],1)
        res = cv.resize(img,None,fx=resize_factor, fy=resize_factor, interpolation = cv.INTER_CUBIC)
        space = 0
        num_boxes = 0
        box_area = [0] *4
        while df['PredictionString'][count].split(" ")[space] != "":
            x = float(df['PredictionString'][count].split(" ")[space+1])*resize_factor
            y = float(df['PredictionString'][count].split(" ")[space+2])*resize_factor
            width = float(df['PredictionString'][count].split(" ")[space+3])*resize_factor
            height = float(df['PredictionString'][count].split(" ")[space+4])*resize_factor
            box_area[num_boxes] = int((width * height) / (resize_factor*1000*resize_factor))
            pts_box = np.array([[x,y],[x,y+height],[x+width,y+height],[x+width,y]], np.int32)
            cv.polylines(res, [pts_box], True, (0,255,0), 3)
            cv.imshow("Test Image",res)
            space +=5
            num_boxes +=1
        num_images += 1

        if num_images<10:
            print("# ",num_images, "| ID: ", patientId,"| Boxes: ", num_boxes, "| Area: ", box_area)
        else:
            print("#",num_images, "| ID: ", patientId,"| Boxes: ", num_boxes, "| Area: ", box_area)
        cv.waitKey(0)
        cv.destroyAllWindows()
