import xml.etree.ElementTree as ET
import os
import caffe
from google.protobuf import text_format
from caffe.proto import caffe_pb2
from caffe.model_libs import *
####### you are supposed to run this script at the CAFFE ROOT ########################

network_name = "ssd"  #your neural net name, need to config
dataset_name = "RMinfantry"#your dataset name, need to config
model_name = "{}_{}".format(network_name,dataset_name)
job_dir = "examples/{}/{}".format(network_name, model_name)
prototxt_dir = "{}/prototxt".format(job_dir)
trainLog_dir = "{}/log".format(job_dir)
trainData_dir = "{}/data".format(job_dir)
trainModel_dir = "{}/model".format(job_dir)

detect_result_file = "{}/result/detect/result.txt".format(job_dir)
annotation_dir = "/home/kangyi/data/RMinfantry/RMinfantry_chassis/Annotations"
labelmap_file = "{}/labelmap_RMinfantry.prototxt".format(trainData_dir)
result_save_dir = "{}/result/test".format(job_dir)
save_result = True

label_pairs=[]
lm_f = open(labelmap_file, 'r')
labelmap = caffe_pb2.LabelMap()
text_format.Merge(str(lm_f.read()), labelmap)
class_num = len(labelmap.item)
for i in range(class_num):
    label_pairs.append([labelmap.item[i].label,labelmap.item[i].name])
lm_f.close()

count=0
gt_bboxs=[]
pre_bbox=[]
TP=[0 for i in range(class_num)]
FP=[0 for i in range(class_num)]
FN=[0 for i in range(class_num)]
TN=[0 for i in range(class_num)]
precision=[0 for i in range(class_num)]
recall=[0 for i in range(class_num)]
accuracy=[0 for i in range(class_num)]
F1=[0 for i in range(class_num)]

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
      

dr_f = open(detect_result_file,'r')
while True:
    line = dr_f.readline()
    if line != '':
        item = line.split()
        if item[0] == "file_name:":
            object_cnt = 0
            del gt_bboxs[:]
            basename = os.path.basename(item[1])
            updateTree = ET.parse("{}/{}.xml".format(annotation_dir,os.path.splitext(basename)[0]))
            root = updateTree.getroot()
            for object in root.findall("object"):
            # bbox list format:
            # xmin ymin xmax ymax label_id label_name matched_flag
                bb_temp = []
                bb_temp.append(int(object.find("bndbox/xmin").text))
                bb_temp.append(int(object.find("bndbox/ymin").text))
                bb_temp.append(int(object.find("bndbox/xmax").text))
                bb_temp.append(int(object.find("bndbox/ymax").text))
                bb_temp.append(get_label_id(object.find("name").text))
                bb_temp.append(object.find("name").text)
                bb_temp.append(0) # a flag, if it is matched
                gt_bboxs.append(bb_temp)
                object_cnt=object_cnt+1
                del bb_temp
            #print(gt_bboxs)
        elif item[0] == "param_name:":
            pass
            #print("parse param")
        elif item[0] == "object:":
            # result.txt file format:
            # xmin ymin xmax ymax label_id confidence label_name
            del pre_bbox[:]
            pre_bbox.append(int(item[1]))
            pre_bbox.append(int(item[2]))
            pre_bbox.append(int(item[3]))
            pre_bbox.append(int(item[4]))
            pre_bbox.append(int(item[5]))#label id
            pre_bbox.append(item[7])#label name
            pre_bbox.append(0)
            for gt_bbox in gt_bboxs:
                if gt_bbox[5] == pre_bbox[5]:
                    iou = computeIOU(gt_bbox,pre_bbox)
                    print("{} iou is : {}".format(gt_bbox[5],iou))
                    if iou>0.5:
                        TP[pre_bbox[4]]=TP[pre_bbox[4]]+1
                        for i in range(class_num):
                            TN[i]=TN[i]+1
                        TN[pre_bbox[4]]=TN[pre_bbox[4]]-1
                        gt_bbox[6] = 1
                        pre_bbox[6] = 1
                        break
            if pre_bbox[6] == 0:
                FP[pre_bbox[4]]=FP[pre_bbox[4]]+1
        elif item[0] == "---":
            count=count+1
            if count%1000 == 0:
                print("Processed {} images...".format(count))
            for gt_bbox in gt_bboxs:
                if gt_bbox[6] == 0:
                    FN[gt_bbox[4]]=FN[gt_bbox[4]]+1
                    for i in range(class_num):
                        TN[i]=TN[i]+1
                    TN[gt_bbox[4]]=TN[gt_bbox[4]]-1
            print("gt_bboxs : {}".format(gt_bboxs))
            print("pre_bbox : {}".format(pre_bbox))
            print("TP : {}".format(TP))
            print("FP : {}".format(FP))
            print("FN : {}".format(FN))
            print("TN : {}".format(TN))
        else:
            print("Incorrect format! item[0] is: {}".format(item[0]))
    else:
        if count%1000 != 0:
            print("Processed {} images...".format(count))
        print("finished!")
        break
dr_f.close()

for i in range(1,class_num):
    if TP[i]+FP[i] != 0:
        precision[i]=float(TP[i])/float(TP[i]+FP[i])
    else:
        precision[i]=0
    if TP[i]+FN[i] != 0:
        recall[i]=float(TP[i])/float(TP[i]+FN[i])
    else:
        recall[i]=0
    if TP[i]+FP[i]+FN[i]+TN[i] != 0:
        accuracy[i] = float(TP[i]+TN[i])/float(TP[i]+FP[i]+FN[i]+TN[i])
    else:
        accuracy[i] = 0
    if precision[i]+recall[i] != 0:
        F1[i]=float(2*precision[i]*recall[i])/float(precision[i]+recall[i])
    else:
        F1[i]=0

    print("{} P is: {}".format(get_label_name(i),precision[i]))
    print("{} R is: {}".format(get_label_name(i),recall[i]))
    print("{} A is: {}".format(get_label_name(i),accuracy[i]))
    print("{} F is: {}".format(get_label_name(i),F1[i]))

if save_result:
    make_if_not_exist(result_save_dir)
    result_save_file = "{}/result.txt".format(result_save_dir)
    result_fp = open(result_save_file, 'w')

    for i in range(1,class_num):
        result_fp.write(label_pairs[i][1])
        result_fp.write(" ")
    result_fp.write("\n")

    result_fp.write("Precision: ")
    for i in range(1,class_num):
        result_fp.write("{}".format(float('%.3f'%precision[i])))
        result_fp.write(" ")
    result_fp.write("\n")

    result_fp.write("Recall: ")
    for i in range(1,class_num):
        result_fp.write("{}".format(float('%.3f'%recall[i])))
        result_fp.write(" ")
    result_fp.write("\n")

    result_fp.write("Accuracy: ")
    for i in range(1,class_num):
        result_fp.write("{}".format(float('%.3f'%accuracy[i])))
        result_fp.write(" ")
    result_fp.write("\n")

    result_fp.write("F1-score: ")
    for i in range(1,class_num):
        result_fp.write("{}".format(float('%.3f'%F1[i])))
        result_fp.write(" ")
    result_fp.write("\n")

    result_fp.close()    
