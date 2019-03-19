


import xml.etree.ElementTree as ET
import numpy as np
import os
import caffe
import argparse
from google.protobuf import text_format
from caffe.proto import caffe_pb2
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

source_file = "{}/result/detect/result.txt".format(job_dir)
annotation_dir = "/home/kangyi/data/VOCdevkit/VOC2007/Annotations"
labelmap_file = "{}/labelmap_voc.prototxt".format(trainData_dir)
result_save_dir = "{}/result/test_loc".format(job_dir)
save_result = True

def get_class_num():
    lm_f = open(labelmap_file, 'r')
    labelmap = caffe_pb2.LabelMap()
    text_format.Merge(str(lm_f.read()), labelmap)
    class_num = len(labelmap.item)
    lm_f.close()
    return class_num

def get_label_id(labelname):
    for i in range(class_num):
        if labelname == label_pairs[i][1]:
            return label_pairs[i][0]
    return -1

def get_label_name(label_id):
    for i in range(class_num):
        if label_id == int(label_pairs[i][0]):
            return label_pairs[i][1]
    return "None"

def computeIOU(box1, box2):
    # box:[xmin,ymin,xmax,ymax]
    in_h = min(box1[3], box2[3]) - max(box1[1], box2[1]) 
    in_w = min(box1[2], box2[2]) - max(box1[0], box2[0]) 
    inter = 0 if in_h<0 or in_w<0 else in_h*in_w 
    union = (box1[3]-box1[1])*(box1[2]-box1[0])+(box2[3]-box2[1])*(box2[2]-box2[0])-inter 
    iou = float(inter) / float(union) 
    return iou

def computeDelta(x1,y1,x2,y2,img_w,img_h):
    a = 0.5*((float(x1-x2)/float(img_w))**2)
    b = 0.5*((float(y1-y2)/float(img_h))**2)
    d = (a+b)**0.5
    return d

if __name__ == "__main__":
    ious = []
    count = 0
    gt_bboxs=[]
    pre_bbox=[]
    label_pairs=[]

    lm_f = open(labelmap_file, 'r')
    labelmap = caffe_pb2.LabelMap()
    text_format.Merge(str(lm_f.read()), labelmap)
    class_num = len(labelmap.item)
    for i in range(class_num):
        label_pairs.append([labelmap.item[i].label,labelmap.item[i].name])
    lm_f.close()

    iou_store = [[0 for i in range(1)] for j in range(class_num)]
    delta_store = [[0 for i in range(1)] for j in range(class_num)]
    iou_avr = [0 for i in range(class_num)]
    delta_avr = [0 for i in range(class_num)]
    C_avr = [0 for i in range(class_num)]

    source_fp = open(source_file,'r')
    while True:
        line = source_fp.readline()
        if line != '':
            item = line.split()
            if item[0] == "file_name:":
                del gt_bboxs[:]
                obj_cnt = 0
                basename = os.path.basename(item[1])
                updateTree = ET.parse("{}/{}.xml".format(annotation_dir,os.path.splitext(basename)[0]))
                root = updateTree.getroot()
                img_w = int(root.find("size/width").text)
                img_h = int(root.find("size/height").text)
                for object in root.findall("object"):
                # bbox list format:
                # xmin ymin xmax ymax label_id label_name x_center y_center img_w img_h matched_flag
                    bb_temp = []
                    bb_temp.append(int(object.find("bndbox/xmin").text))# 0
                    bb_temp.append(int(object.find("bndbox/ymin").text))# 1
                    bb_temp.append(int(object.find("bndbox/xmax").text))# 2
                    bb_temp.append(int(object.find("bndbox/ymax").text))# 3
                    bb_temp.append(get_label_id(object.find("name").text))# 4
                    bb_temp.append(object.find("name").text)# 5
                    bb_temp.append((bb_temp[0]+bb_temp[2])/2.0)# 6 x center
                    bb_temp.append((bb_temp[1]+bb_temp[3])/2.0)# 7 y center
                    bb_temp.append(img_w)# 8
                    bb_temp.append(img_h)# 9
                    bb_temp.append(0)# 10 matched flag
                    bb_temp.append(obj_cnt)# 11 object counter
                    gt_bboxs.append(bb_temp)
                    obj_cnt += 1
                    del bb_temp
                #print("gt_bbox is : {}".format(gt_bboxs))
            elif item[0] == "param_name:":
                pass
                #print("parse param")
            elif item[0] == "object:":
                # result.txt file format:
                # xmin ymin xmax ymax label_id confidence label_name
                del ious[:]
                del pre_bbox[:]
                pre_bbox.append(int(item[1]))# 0
                pre_bbox.append(int(item[2]))# 1
                pre_bbox.append(int(item[3]))# 2
                pre_bbox.append(int(item[4]))# 3
                pre_bbox.append(int(item[5]))# 4 label id
                pre_bbox.append(item[7])# 5 label name
                pre_bbox.append((pre_bbox[0]+pre_bbox[2])/2.0)# 6 x center
                pre_bbox.append((pre_bbox[1]+pre_bbox[3])/2.0)# 7 y center
                for gt_bbox in gt_bboxs:
                    if gt_bbox[5] == pre_bbox[5] and gt_bbox[10] == 0:
                        iou = computeIOU(gt_bbox,pre_bbox)
                        ious.append([iou,gt_bbox[11]])
                        del iou
                #print("ious is : {}".format(ious))
                if len(ious)>0:
                    ious.sort(key=lambda i : i[0])
                    iou_store[pre_bbox[4]].append(ious[-1][0])
                    gt_bboxs[ious[-1][1]][10] = 1
                    x1=pre_bbox[6]
                    y1=pre_bbox[7]
                    x2=gt_bboxs[ious[-1][1]][6]
                    y2=gt_bboxs[ious[-1][1]][7]
                    w=gt_bboxs[ious[-1][1]][8]
                    h=gt_bboxs[ious[-1][1]][9]
                    delta = computeDelta(x1,y1,x2,y2,w,h)
                    delta_store[pre_bbox[4]].append(delta)
                    del delta
            elif item[0] == "---":
                #print("gt_bbox is : {}".format(gt_bboxs))
                count += 1
                if count%1000 == 0:
                    print("Processed {} images...".format(count))
            else:
                print("Incorrect format !")
        else:
            #print("iou_store is : {}".format(iou_store))
            print("finished !")
            break
    source_fp.close()

    for i in range(1,class_num):
        data_len = len(iou_store[i])
        iou_sum = 0
        delta_sum = 0
        for j in range(1,data_len):
            iou_sum += iou_store[i][j]
            delta_sum += delta_store[i][j]
        iou_avr[i] = iou_sum/float(data_len-1)
        delta_avr[i] = delta_sum/float(data_len-1)
        C_avr[i] = iou_avr[i]**2*(1-delta_avr[i])

    print("iou avr is : {}".format(iou_avr))
    print("delta avr is : {}".format(delta_avr))

    if save_result:
        make_if_not_exist(result_save_dir)
        result_save_file = "{}/result.txt".format(result_save_dir)
        result_fp = open(result_save_file, 'w')

        result_fp.write("Class: ")
        for i in range(1,class_num):
            result_fp.write(label_pairs[i][1])
            result_fp.write(" ")
        result_fp.write("\n")

        result_fp.write("IoU_average: ")
        for i in range(1,class_num):
            result_fp.write("{}".format(float('%.3f'%iou_avr[i])))
            result_fp.write(" ")
        result_fp.write("\n")

        result_fp.write("delta_average: ")
        for i in range(1,class_num):
            result_fp.write("{}".format(float('%.3f'%delta_avr[i])))
            result_fp.write(" ")
        result_fp.write("\n")

        result_fp.write("C_average: ")
        for i in range(1,class_num):
            result_fp.write("{}".format(float('%.3f'%C_avr[i])))
            result_fp.write(" ")
        result_fp.write("\n")

        result_fp.close()

