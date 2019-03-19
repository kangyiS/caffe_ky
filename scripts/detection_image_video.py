import numpy as np
import matplotlib.pyplot as plt
import pylab
import os
import caffe
import sys
import pickle
import cv2
import time
from datetime import datetime

caffe_root = os.getcwd()
deploy = 'models/VGGNet/VOC0712/SSD_RMinfantry_300x300/deploy.prototxt'
weight = 'models/VGGNet/VOC0712/SSD_RMinfantry_300x300/VGG_SSD_RMinfantry_300x300_iter_20000.caffemodel'
video_full_path = '/home/kangyi/caffe/examples/videos/RMinfantry_videos/video6.mp4'
testImg = '/home/kangyi/data/RMinfantry/JPEGImages/16747.jpg'
image_size = 300

def net_init():
    print
    'initilize ... '
    sys.path.insert(0, caffe_root + 'python')
    caffe.set_mode_gpu()
    caffe.set_device(0)
    net = caffe.Net(deploy, weight, caffe.TEST)
    return net


def testImage(image, net):
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    # changing blob from H*W*C to C*H*W
    transformer.set_transpose('data', (2, 0, 1))
    # ensure the pixel scale is range from (0,255)
    transformer.set_raw_scale('data', 255)
    # change channel order from RGB to BGR
    transformer.set_channel_swap('data', (2, 1, 0))
    # reshape data
    net.blobs['data'].reshape(1, 3, image_size, image_size)
    # input data and preprocess
    if isinstance(image, str):
        net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(image))
    else:
        net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image_from_frame(image))
    # testing model is just a forward process
    out = net.forward()
    xmin = out['detection_out'][0][0][0][3]
    ymin = out['detection_out'][0][0][0][4]
    xmax = out['detection_out'][0][0][0][5]
    ymax = out['detection_out'][0][0][0][6]
    xcenter = (xmin + xmax)/2
    ycenter = (ymin + ymax)/2
    print(out)
    print(xcenter)
    print(ycenter)
    print(datetime.now())
    print('-------------------------------------------')
    #print(net.blobs.keys())
    '''
    filters = net.params['conv1_1'][0].data
    with open('FirstLayerFilter.pickle', 'wb') as f:
        pickle.dump(filters, f)
    vis_square(filters.transpose(0, 2, 3, 1))
  

    feat = net.blobs['conv1_1'].data[0, :64]
    #with open('FirstLayerOutput.pickle', 'wb') as f:
    #    pickle.dump(feat, f)
    vis_square(feat, padval=1)

    feat = net.blobs['conv1_2'].data[0, :64]
    vis_square(feat, padval=1)
    pool = net.blobs['pool1'].data[0, :64]
    vis_square(pool, padval=1)
    feat = net.blobs['conv2_1'].data[0, :128]
    vis_square(feat, padval=1)

'''

def vis_square(data, padsize=1, padval=0):
    data -= data.min()
    data /= data.max()
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    plt.imshow(data)
    pylab.show()
    print
    data.shape

'''
if __name__ == "__main__":
    net = net_init()
    cap = cv2.VideoCapture(video_full_path)
    if(cap.isOpened()):
        print('Read the video successfully')
        frame_count = 1
        success = True
        while (success):
            success, frame = cap.read()  # frame is a BGR image
            frameRGB = frame[:, :, (2, 1, 0)]
            print('-----------------------------------Read a new frame: ' + str(success) + '  ' + str(frame_count))
            getNetDetails(frameRGB, net)  # the input needs a RGB image
            #cv2.imshow('video', frame)

            frame_count = frame_count + 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
    else:
        print('Can not read the video')
'''
if __name__ == "__main__":
    net = net_init()
    testImage(testImg, net)  # the input needs a RGB image
    #cv2.imshow('video', frame)
