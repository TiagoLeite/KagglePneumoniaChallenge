import cv2 as cv
import numpy as np
import argparse
from matplotlib import pyplot as plt
import glob, pylab, pandas as pd
import os

PATH_TO_TEST_IMAGES_DIR = 'data/images/train_jpg'
df = pd.read_csv('data/train_labels.csv')

# ADJUST THESE PARAMETERS
resize_factor = 0.5
number_images = 100

# VARIABLES
img_count = 0
box_area = [0]*4
num_boxes = 0
pos_box = 0

for count in range(number_images):
    patientId = df['patientId'][count]

    if count==0 and (list(df['patientId'][count])==list(df['patientId'][count+1])):
        TEST_IMAGE_PATHS = glob.glob(PATH_TO_TEST_IMAGES_DIR+'/'+patientId+'.jpg')
        img = cv.imread(TEST_IMAGE_PATHS[0],1)
        res = cv.resize(img,None,fx=resize_factor, fy=resize_factor, interpolation = cv.INTER_CUBIC)

        x = float(df['x'][count]*resize_factor)
        y = float(df['y'][count]*resize_factor)
        width = float(df['width'][count]*resize_factor)
        height = float(df['height'][count]*resize_factor)
        box_area[pos_box] = int((width * height) / (resize_factor*1000*resize_factor))
        num_boxes +=1
        pos_box +=1
        pts_box_0 = np.array([[x,y],[x,y+height],[x+width,y+height],[x+width,y]], np.int32)
        cv.polylines(res, [pts_box_0], True, (0,0,255), 3)
        cv.imshow("Test Image",res)


    elif count==0 and (list(df['patientId'][count])!=list(df['patientId'][count+1])):
        TEST_IMAGE_PATHS = glob.glob(PATH_TO_TEST_IMAGES_DIR+'/'+patientId+'.jpg')
        img = cv.imread(TEST_IMAGE_PATHS[0],1)
        res = cv.resize(img,None,fx=resize_factor, fy=resize_factor, interpolation = cv.INTER_CUBIC)

        x = float(df['x'][count]*resize_factor)
        y = float(df['y'][count]*resize_factor)
        width = float(df['width'][count]*resize_factor)
        height = float(df['height'][count]*resize_factor)
        box_area[pos_box] = int((width * height) / (resize_factor*1000*resize_factor))
        num_boxes +=1
        if img_count<10:
            print("# ",img_count, "| ID: ", patientId,"| Boxes: ", num_boxes, "| Area: ", box_area)
        else:
            print("#",img_count, "| ID: ", patientId,"| Boxes: ", num_boxes, "| Area: ", box_area)
        pts_box_0 = np.array([[x,y],[x,y+height],[x+width,y+height],[x+width,y]], np.int32)
        cv.polylines(res, [pts_box_0], True, (0,0,255), 3)
        cv.imshow("Test Image",res)
        cv.waitKey(0)
        num_boxes = 0
        pos_box = 0
        img_count += 1

    elif list(df['patientId'][count])!=list(df['patientId'][count-1]) and list(df['patientId'][count])!=list(df['patientId'][count+1]):
        TEST_IMAGE_PATHS = glob.glob(PATH_TO_TEST_IMAGES_DIR+'/'+patientId+'.jpg')
        img = cv.imread(TEST_IMAGE_PATHS[0],1)
        res = cv.resize(img,None,fx=resize_factor, fy=resize_factor, interpolation = cv.INTER_CUBIC)

        x = float(df['x'][count]*resize_factor)
        y = float(df['y'][count]*resize_factor)
        width = float(df['width'][count]*resize_factor)
        height = float(df['height'][count]*resize_factor)
        box_area[pos_box] = int((width * height) / (resize_factor*1000*resize_factor))
        num_boxes +=1
        if img_count<10:
            print("# ",img_count, "| ID: ", patientId,"| Boxes: ", num_boxes, "| Area: ", box_area)
        else:
            print("#",img_count, "| ID: ", patientId,"| Boxes: ", num_boxes, "| Area: ", box_area)
        pts_box = np.array([[x,y],[x,y+height],[x+width,y+height],[x+width,y]], np.int32)
        cv.polylines(res, [pts_box], True,(0,0,255), 3)
        cv.imshow("Test Image",res)
        cv.waitKey(0)
        num_boxes = 0
        pos_box = 0
        img_count += 1

    elif list(df['patientId'][count])==list(df['patientId'][count-1]) and list(df['patientId'][count])!=list(df['patientId'][count+1]):
        x = float(df['x'][count]*resize_factor)
        y = float(df['y'][count]*resize_factor)
        width = float(df['width'][count]*resize_factor)
        height = float(df['height'][count]*resize_factor)
        box_area[pos_box] = int((width * height) / (resize_factor*1000*resize_factor))
        num_boxes +=1
        pos_box +=1
        if img_count<10:
            print("# ",img_count, "| ID: ", patientId,"| Boxes: ", num_boxes, "| Area: ", box_area)
        else:
            print("#",img_count, "| ID: ", patientId,"| Boxes: ", num_boxes, "| Area: ", box_area)
        pts_box = np.array([[x,y],[x,y+height],[x+width,y+height],[x+width,y]], np.int32)
        cv.polylines(res, [pts_box], True,(0,0,255), 3)
        cv.imshow("Test Image",res)
        cv.waitKey(0)
        num_boxes = 0
        pos_box = 0
        img_count += 1

    elif list(df['patientId'][count])!=list(df['patientId'][count-1]) and list(df['patientId'][count])==list(df['patientId'][count+1]):
        TEST_IMAGE_PATHS = glob.glob(PATH_TO_TEST_IMAGES_DIR+'/'+patientId+'.jpg')
        img = cv.imread(TEST_IMAGE_PATHS[0],1)
        res = cv.resize(img,None,fx=resize_factor, fy=resize_factor, interpolation = cv.INTER_CUBIC)

        x = float(df['x'][count]*resize_factor)
        y = float(df['y'][count]*resize_factor)
        width = float(df['width'][count]*resize_factor)
        height = float(df['height'][count]*resize_factor)
        box_area[pos_box] = int((width * height) / (resize_factor*1000*resize_factor))
        num_boxes +=1
        pos_box +=1
        pts_box = np.array([[x,y],[x,y+height],[x+width,y+height],[x+width,y]], np.int32)
        cv.polylines(res, [pts_box], True,(0,0,255), 3)
        cv.imshow("Test Image",res)

    elif list(df['patientId'][count])==list(df['patientId'][count-1]) and list(df['patientId'][count])==list(df['patientId'][count+1]):
        x = float(df['x'][count]*resize_factor)
        y = float(df['y'][count]*resize_factor)
        width = float(df['width'][count]*resize_factor)
        height = float(df['height'][count]*resize_factor)
        box_area[pos_box] = int((width * height) / (resize_factor*1000*resize_factor))
        num_boxes +=1
        pos_box +=1
        pts_box = np.array([[x,y],[x,y+height],[x+width,y+height],[x+width,y]], np.int32)
        cv.polylines(res, [pts_box], True,(0,0,255), 3)
        cv.imshow("Test Image",res)
