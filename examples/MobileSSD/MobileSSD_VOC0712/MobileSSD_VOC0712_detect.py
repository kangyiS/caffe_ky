'''
you can use this script detect one image, a group of images, a video file or the video strame from a camera

compute speed,using VOC2007 dataset:
video:     58fps(forward), 20fps(loop)
one image: 36fps(forward), 34fps(loop)
images:    73fps(forward), 55fps(loop)

'''
import os
import os.path
import sys
import cv2
import time
import numpy as np
# Make sure that caffe is on the python path:
caffe_root = './'
os.chdir(caffe_root)
sys.path.insert(0, os.path.join(caffe_root, 'python'))
import caffe

from google.protobuf import text_format
from caffe.proto import caffe_pb2
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
img_save_dir = "{}/result/detect".format(job_dir)

# the caffemodel you want to use
model_weights = "{}/MobileNetSSD2000_deploy.caffemodel".format(trainModel_dir)#need to config

deploy_net_file = "{}/MobileNetSSD_deploy.prototxt".format(prototxt_dir)

# used to show label name
labelmap_file = "{}/labelmap_voc.prototxt".format(trainData_dir)
# image_resize has to be matched to the input size of your network
image_resize = 300 #need to config

# source_file can be an image file, a directory, a video file or the camera id
source_file = "/home/kangyi/data/VOCdevkit/VOC2007/JPEGImages/000001.jpg"
#source_file = "/home/kangyi/data/VOCdevkit/VOC2007/JPEGImages"
#source_file = "/home/kangyi/caffe/examples/videos/ILSVRC2015_train_00755001.mp4"
#source_file = 0 # use camera
file_type = "image" # the type of source_file : "video" or "image"
save_img = True # save images or video
gpu_id = 0

def get_labelname(labelmap, labels):
    num_labels = len(labelmap.item)
    labelnames = []
    if type(labels) is not list:
        labels = [labels]
    for label in labels:
        found = False
        for i in xrange(0, num_labels):
            if label == labelmap.item[i].label:
                found = True
                labelnames.append(labelmap.item[i].display_name)
                break
        assert found == True
    return labelnames

class CaffeDetection:
    def __init__(self, gpu_id, model_def, model_weights, image_resize, labelmap_file):
        caffe.set_device(gpu_id)
        caffe.set_mode_gpu()

        self.image_resize = image_resize
        # Load the net in the test phase for inference, and configure input preprocessing.
        self.net = caffe.Net(model_def,      # defines the structure of the model
                             model_weights,  # contains the trained weights
                             caffe.TEST)     # use test mode (e.g., don't perform dropout)
         # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
        self.transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
        self.transformer.set_transpose('data', (2, 0, 1))
        self.transformer.set_mean('data', np.array([104, 117, 123])) # mean pixel
        # the reference model operates on images in [0,255] range instead of [0,1]
        self.transformer.set_raw_scale('data', 255)
        # the reference model has channels in BGR order instead of RGB
        self.transformer.set_channel_swap('data', (2, 1, 0))

        # load PASCAL VOC labels
        file = open(labelmap_file, 'r')
        self.labelmap = caffe_pb2.LabelMap()
        text_format.Merge(str(file.read()), self.labelmap)

    def detect(self, image_file, conf_thresh=0.5, topn=5):
        self.net.blobs['data'].reshape(1, 3, self.image_resize, self.image_resize)
        if isinstance(image_file, str):
            image = caffe.io.load_image(image_file)
        else:
            image = caffe.io.load_image_from_frame(image_file)

        transformed_image = self.transformer.preprocess('data', image)
        self.net.blobs['data'].data[...] = transformed_image

        # Forward pass.
        forward_start_time = time.time()
        detections = self.net.forward()['detection_out']
        forward_time = time.time()-forward_start_time
        forward_fps = float('%.1f'%(1.0/forward_time))
        #print("Forward speed is : {} fps".format(forward_fps))
        # Parse the outputs.
        det_label = detections[0,0,:,1]
        det_conf = detections[0,0,:,2]
        det_xmin = detections[0,0,:,3]
        det_ymin = detections[0,0,:,4]
        det_xmax = detections[0,0,:,5]
        det_ymax = detections[0,0,:,6]

        # Get detections with confidence higher than 0.5.
        top_indices = [i for i, conf in enumerate(det_conf) if conf >= conf_thresh]

        top_conf = det_conf[top_indices]
        top_label_indices = det_label[top_indices].tolist()
        top_labels = get_labelname(self.labelmap, top_label_indices)
        top_xmin = det_xmin[top_indices]
        top_ymin = det_ymin[top_indices]
        top_xmax = det_xmax[top_indices]
        top_ymax = det_ymax[top_indices]

        result = []
        for i in xrange(min(topn, top_conf.shape[0])):
            xmin = top_xmin[i] # xmin = int(round(top_xmin[i] * image.shape[1]))
            ymin = top_ymin[i] # ymin = int(round(top_ymin[i] * image.shape[0]))
            xmax = top_xmax[i] # xmax = int(round(top_xmax[i] * image.shape[1]))
            ymax = top_ymax[i] # ymax = int(round(top_ymax[i] * image.shape[0]))
            score = top_conf[i]
            label = int(top_label_indices[i])
            label_name = top_labels[i]
            result.append([xmin, ymin, xmax, ymax, label, score, label_name])
        return result

detection = CaffeDetection(gpu_id,deploy_net_file,model_weights,image_resize,labelmap_file)
make_if_not_exist(img_save_dir)

if os.path.isfile(source_file) and file_type is "image": # detect one image
    result_file = "{}/result.txt".format(img_save_dir)
    if os.path.exists(result_file) and save_img:
        os.remove(result_file)
    start_time = time.time()
    result = detection.detect(source_file)
    if save_img:
        img_save_file = "{}/{}".format(img_save_dir,os.path.basename(source_file))
        img = cv2.imread(source_file)
        height = img.shape[0]
        width = img.shape[1]
        result_fp = open(result_file, 'w')
        result_fp.write("file_name: {} \n".format(source_file))
        result_fp.write("param_name: xmin ymin xmax ymax label_id confidence label_name \n")
        for item in result:
            xmin = int(round(item[0] * width))
            ymin = int(round(item[1] * height))
            xmax = int(round(item[2] * width))
            ymax = int(round(item[3] * height))
            cv2.rectangle(img, (xmin,ymin), (xmax, ymax), (255,0,0), 5)
            cv2.putText(img, "{} {}".format(item[-1], float('%.3f'%item[-2])),(xmin,ymin+10),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),1)
            result_fp.write("object: {} {} {} {} {} {} {} \n".format(xmin,ymin,xmax,ymax,item[4],item[5],item[6]))
        result_fp.write("--- \n")
        cv2.imwrite(img_save_file,img)
        print("Save the image to {}".format(img_save_file))
        result_fp.close()
        print("Save result file to {}".format(result_file))
    print("It takes {} s...".format(time.time()-start_time))
elif (os.path.isfile(source_file) or source_file is 0) and file_type is "video": # detect a video
    cap = cv2.VideoCapture(source_file)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    size = (int(cap.get(3)), int(cap.get(4))) # 3 means width, 4 means height
    fps = 30 # fps of a video
    outVideo = cv2.VideoWriter("{}/result.avi".format(img_save_dir),fourcc,fps,size) 
    if cap.isOpened():
        print("Read the video successfully!")
        success = True
        start_time = time.time()
        loop_start_time = time.time()
        while success:
            success, frame = cap.read()
            if success:
                frameRGB = frame[:,:,(2,1,0)]
                result = detection.detect(frameRGB)
                loop_time = time.time()-loop_start_time
                loop_fps = float('%.1f' % (1.0/loop_time))
                loop_start_time = time.time()
                print("Video running time : {} s and speed : {} fps".format(float('%.3f'%(time.time()-start_time)), loop_fps))                
                height = frame.shape[0]
                width = frame.shape[1]
                for item in result:
                    xmin = int(round(item[0] * width))
                    ymin = int(round(item[1] * height))
                    xmax = int(round(item[2] * width))
                    ymax = int(round(item[3] * height))
                    cv2.rectangle(frame, (xmin,ymin), (xmax, ymax), (255,0,0), 5)
                    cv2.putText(frame, "{} {}".format(item[-1], float('%.3f'%item[-2])),(xmin,ymin+10),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),1)
                    cv2.putText(frame, "{} FPS".format(loop_fps),(0,30),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),1)
                cv2.imshow("video", frame)
                if cv2.waitKey(1) & 0xff == ord('q'):
                    cv2.destroyWindow('video')
                    cap.release()
                    break;
                if save_img:
                    outVideo.write(frame)
    else:
        print("Can not read the video!")
    cap.release()
    outVideo.release()
elif os.path.isdir(source_file): # detect a group of images
    count=0
    result_file = "{}/result.txt".format(img_save_dir)
    if os.path.exists(result_file) and save_img:
        os.remove(result_file)
    start_time = time.time()
    for root, dirs, files in os.walk(source_file):
        for file in files:
            if os.path.splitext(file)[1] == '.jpg':                
                count=count+1
                img_file = "{}/{}".format(source_file, file)
                result = detection.detect(img_file)
                if count % 1000 == 0:
                    print("Processed {} images... it takes {} s".format(count, time.time()-start_time))
                    start_time = time.time()
                if save_img:
                    img_save_file = "{}/{}".format(img_save_dir,file)
                    img = cv2.imread(img_file)
                    height = img.shape[0]
                    width = img.shape[1]
                    result_fp = open(result_file, 'a+')
                    result_fp.write("file_name: {}/{} \n".format(source_file,file))
                    result_fp.write("param_name: xmin ymin xmax ymax label_id confidence label_name \n")            
                    for item in result:
                        xmin = int(round(item[0] * width))
                        ymin = int(round(item[1] * height))
                        xmax = int(round(item[2] * width))
                        ymax = int(round(item[3] * height))
                        cv2.rectangle(img, (xmin,ymin), (xmax, ymax), (255,0,0), 5)
                        cv2.putText(img, "{} {}".format(item[-1], float('%.3f'%item[-2])),(xmin,ymin+10),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),1)
                        result_fp.write("object: {} {} {} {} {} {} {} \n".format(xmin,ymin,xmax,ymax,item[4],item[5],item[6]))
                    result_fp.write("--- \n")
                    cv2.imwrite(img_save_file,img)
                    #print("Save the image to {}".format(img_save_file))
                    result_fp.close()
                    #print("Save result file to {}".format(result_file))
    if count % 1000 is not 0:
        print("Processed {} images... it takes {} s".format(count, time.time()-start_time))
else: # incorrect source file
    print("Unknown source type... ")
