'''
Author: Mr.K

You can use this script to plot the P-R curve.
Before running this script, you need to run the test script to get a test result file. 
The format of the test result file is as followed:

Class:     *** *** *** ... *** *** ***
Precision: *** *** *** ... *** *** ***
Recall:    *** *** *** ... *** *** ***
Accuracy:  *** *** *** ... *** *** ***
F1-score:  *** *** *** ... *** *** ***

This plot script just get precision values and recall values from the test result file.
After running this script, you will get many P-R curve images that it depend on the class number.
Also, you will get a plot result file.
There are the points which are used during ploting curves in the result file.
The format of the plot result file is as followed:

average Precision: *** *** *** ... *** *** ***
average Recall:    *** *** *** ... *** *** ***
average AP:        ***

(class name) Precision: *** *** *** ... *** *** ***
(class name) Recall:    *** *** *** ... *** *** ***
(class name) AP:        ***
            .
            .
            .
(class name) Precision: *** *** *** ... *** *** ***
(class name) Recall:    *** *** *** ... *** *** ***
(class name) AP:        ***

At the beginning of the file, there are "average Precision", "average Recall" and "average AP"(mAP).
The "average Precision" is the average for the precisions of all the class, the "average Recall" is the same.

In this file, AP means average precision, it's the area below the P-R curve.
So "average AP" is usually called mean AP(mAP).

'''

import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
import ssd_VOC0712_test as test
from caffe.model_libs import *

####### you are supposed to run this script at the CAFFE ROOT ########################

network_name = "ssd"  #your neural net name, need to config
dataset_name = "VOC0712"#your dataset name, need to config
model_name = "{}_{}".format(network_name,dataset_name)
job_dir = "examples/{}/{}".format(network_name, model_name)
prototxt_dir = "{}/prototxt".format(job_dir)
trainLog_dir = "{}/log".format(job_dir)
trainData_dir = "{}/data".format(job_dir)
trainModel_dir = "{}/model".format(job_dir)
result_dir = "{}/result".format(job_dir)

img_save_dir = "{}/PRcurve".format(result_dir)

confidence_list = np.arange(0,1,0.1)
count=0

plt_data = []
labelname = []
class_num = test.get_class_num()

precision = []
recall = []
accuracy = []
F1 = []

make_if_not_exist(img_save_dir)

for i in range(class_num):
    temp0_data = []
    temp1_data = []
    temp2_data = []
    for j in range(len(confidence_list)):
        temp1_data.append(0.0)#recall
        temp2_data.append(0.0)#precision
    temp0_data.append(temp1_data)
    temp0_data.append(temp2_data)
    plt_data.append(temp0_data)
    labelname.append(0)
    del temp0_data
    del temp1_data
    del temp2_data
labelname[0]="average"

#os.system("python examples/{}/{}/{}_detect.py --confidence=0.01".format(network_name,model_name,model_name))

for conf in confidence_list:
    print("conf is {}".format(conf))
    os.system("python examples/{}/{}/{}_test.py --confidence={}".format(network_name,model_name,model_name,conf))
    shutil.copyfile("{}/test/result.txt".format(result_dir),"{}/test/result_{}.txt".format(result_dir,count))
    temp_fp = open("{}/test/result.txt".format(result_dir),'r')
    while True:
        line = temp_fp.readline()
        if line != '':
            item = line.split()
            if item[0] == "Class:":
                for i in range(1,class_num):
                    labelname[i]=item[i]
            elif item[0] == "Precision:":
                for i in range(1,class_num):
                    plt_data[i][1][count]=float(item[i])
            elif item[0] == "Recall:":
                for i in range(1,class_num):
                    plt_data[i][0][count]=float(item[i])
            else:
                pass
        else:
            print("conf {} has finished !".format(conf))
            break
    temp_fp.close()
    count += 1


for i in range(count):
    sum_r=0.0
    sum_p=0.0
    for j in range(1,class_num):
        sum_r += plt_data[j][0][i]
        sum_p += plt_data[j][1][i]
    plt_data[0][0][i]=sum_r/float(class_num-1)
    plt_data[0][1][i]=sum_p/float(class_num-1)

area = [0 for i in range(class_num)]
for i in range(class_num):
    for j in range(count-1):
        r1 = plt_data[i][0][j]
        r2 = plt_data[i][0][j+1]
        p1 = plt_data[i][1][j]
        p2 = plt_data[i][1][j+1]
        area[i] += (r1-r2)*((p2-p1)/2.0+p1)
    area[i] += plt_data[i][0][count-1]*plt_data[i][1][count-1]

#************************ save plot data ***********************#
result_fp = open("{}/result.txt".format(img_save_dir),'w')

for i in range(class_num):
    result_fp.write("{} Precision: ".format(labelname[i]))
    for j in range(count):
        result_fp.write(str(float('%.3f'%plt_data[i][1][j])))
        result_fp.write(" ")
    result_fp.write("\n")

    result_fp.write("{} Recall: ".format(labelname[i]))
    for j in range(count):
        result_fp.write(str(float('%.3f'%plt_data[i][0][j])))
        result_fp.write(" ")
    result_fp.write("\n")

    result_fp.write("{} AP: ".format(labelname[i]))
    result_fp.write(str(float('%.3f'%area[i])))
    result_fp.write("\n")

result_fp.close()

#*********************** save P-R curve images ***************#
for i in range(class_num):
    plt.cla()
    plt.plot(plt_data[i][0],plt_data[i][1])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("P-R curve for {}".format(labelname[i]))
    #plt.show()
    plt.savefig("{}/pr_curve_{}.png".format(img_save_dir, labelname[i]))

