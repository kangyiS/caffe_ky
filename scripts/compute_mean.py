'''
Author:       kangyi
Introduction: This script is used to compute mean of .jpg images
              It traverses every pixel to compute mean, but this method takes too much time. We need to find a batter way.
'''
import cv2
import os
import numpy as np

JPGfile_dir = '/home/kangyi/data/RMinfantry/JPEGImages'
imageList_all = []
sum_oneImage = {'b': 0, 'g': 0, 'r': 0}
mean_oneImage = {'b': 0, 'g': 0, 'r': 0}
sum_allImages = {'b': 0, 'g': 0, 'r': 0}
mean_allImages = {'b': 0, 'g': 0, 'r': 0}

for root, dirs, files in os.walk(JPGfile_dir):
    for file in files:
        if os.path.splitext(file)[1] == '.jpg':
            imageList_all.append(file)
imageNum = imageList_all.__len__()
for image_index in range(imageNum):
    image = cv2.imread('{}/{}'.format(JPGfile_dir, imageList_all[image_index]))
    image_height  = image.shape[0]
    image_width   = image.shape[1]
    image_channel = image.shape[2]
    sum_oneImage['b'] = 0
    sum_oneImage['g'] = 0
    sum_oneImage['r'] = 0
    if image_channel == 3:
        b, g, r = cv2.split(image)
        for i in range(image_height):
            for j in range(image_width):
                sum_oneImage['b'] += b[i][j]
                sum_oneImage['g'] += g[i][j]
                sum_oneImage['r'] += r[i][j]
        mean_oneImage['b'] = sum_oneImage['b']/image_height/image_width
        mean_oneImage['g'] = sum_oneImage['g']/image_height/image_width
        mean_oneImage['r'] = sum_oneImage['r']/image_height/image_width
    sum_allImages['b'] += mean_oneImage['b']
    sum_allImages['g'] += mean_oneImage['g']
    sum_allImages['r'] += mean_oneImage['r']
    print('{}:{}--{}'.format(image_index, imageNum, mean_oneImage))
mean_allImages['b'] = sum_allImages['b'] / imageNum
mean_allImages['g'] = sum_allImages['g'] / imageNum
mean_allImages['r'] = sum_allImages['r'] / imageNum
print('all--{}'.format(mean_allImages))
