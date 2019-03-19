import os
import matplotlib.pyplot as plt
from caffe.model_libs import *

####### you are supposed to run this script at the CAFFE ROOT ########################

network_name = "MobileSSD"  #your neural net name, need to config
dataset_name = "VOC0712"#your dataset name, need to config
model_name = "{}_{}".format(network_name,dataset_name)
job_dir = "examples/{}/{}".format(network_name, model_name)
prototxt_dir = "{}/prototxt".format(job_dir)
trainLog_dir = "{}/log".format(job_dir)
trainData_dir = "{}/data".format(job_dir)
trainModel_dir = "{}/model".format(job_dir)
result_dir = "{}/result".format(job_dir)

source1_file = "examples/ssd/ssd_VOC0712/result/PRcurve/result.txt"
source2_file = "examples/MobileSSD/MobileSSD_VOC0712/result/PRcurve/result.txt"
img_save_dir = "{}/PRcurve/compare".format(result_dir)

class_num = 21
data_len = 10
make_if_not_exist(img_save_dir)

label = {"average":0, "aeroplane":1, "bicycle":2, "bird":3, "boat":4, "bottle":5,
         "bus":6, "car":7, "cat":8, "chair":9, "cow":10, "diningtable":11,"dog":12,
          "horse":13, "motorbike":14, "person":15, "pottedplant":16, "sheep":17, "sofa":18, "train":19, "tvmonitor":20}
labelname = ["average","aeroplane","bicycle","bird","boat","bottle","bus","car","cat","chair","cow",
             "diningtable","dog","horse","motorbike","person","pottedplant","sheep","sofa","train","tvmonitor"]


plt_data1 = []
for i in range(class_num):
    temp0_data = []
    temp1_data = []
    temp2_data = []
    for j in range(data_len):
        temp1_data.append(0.0)#recall
        temp2_data.append(0.0)#precision
    temp0_data.append(temp1_data)
    temp0_data.append(temp2_data)
    plt_data1.append(temp0_data)
    del temp0_data
    del temp1_data
    del temp2_data

plt_data2 = []
for i in range(class_num):
    temp0_data = []
    temp1_data = []
    temp2_data = []
    for j in range(data_len):
        temp1_data.append(0.0)#recall
        temp2_data.append(0.0)#precision
    temp0_data.append(temp1_data)
    temp0_data.append(temp2_data)
    plt_data2.append(temp0_data)
    del temp0_data
    del temp1_data
    del temp2_data

source1_fp = open(source1_file,'r')
while True:
    line = source1_fp.readline()
    if line != '':
        item = line.split()
        if item[1] == "Precision:":
            for i in range(data_len):
                plt_data1[label[item[0]]][1][i] = float(item[i+2])
        elif item[1] == "Recall:":
            for i in range(data_len):
                plt_data1[label[item[0]]][0][i] = float(item[i+2])
        else:
            pass
    else:
        print("source1 is done !")
        break
source1_fp.close()

source2_fp = open(source2_file,'r')
while True:
    line = source2_fp.readline()
    if line != '':
        item = line.split()
        if item[1] == "Precision:":
            for i in range(data_len):
                plt_data2[label[item[0]]][1][i] = float(item[i+2])
        elif item[1] == "Recall:":
            for i in range(data_len):
                plt_data2[label[item[0]]][0][i] = float(item[i+2])
        else:
            pass
    else:
        print("source2 is done !")
        break
source2_fp.close()

for i in range(class_num):
    plt.cla()
    plt.plot(plt_data1[i][0],plt_data1[i][1],color='red',linestyle="-",label="SSD")
    plt.plot(plt_data2[i][0],plt_data2[i][1],color='blue',linestyle="--",label="fastSSD")
    plt.legend()
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("P-R curve for {}".format(labelname[i]))
    plt.savefig("{}/pr_curve_{}.png".format(img_save_dir, labelname[i]))
    
