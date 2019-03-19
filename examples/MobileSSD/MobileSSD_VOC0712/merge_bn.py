import numpy as np  
import sys,os  
caffe_root = '/home/kangyi/caffe/'
sys.path.insert(0, caffe_root + 'python')  
import caffe  

train_proto = 'examples/MobileSSD/MobileSSD_VOC0712/prototxt/MobileNetSSD_train.prototxt'  
train_model = 'examples/MobileSSD/MobileSSD_VOC0712/model/MobileSSD_VOC0712_iter_7000.caffemodel'  #should be your snapshot caffemodel

deploy_proto = 'examples/MobileSSD/MobileSSD_VOC0712/prototxt/MobileNetSSD_deploy.prototxt'  
save_model = 'examples/MobileSSD/MobileSSD_VOC0712/MobileNetSSD7000_deploy.caffemodel'

def merge_bn(train_net, deploy_net):
    '''
    merge the batchnorm, scale layer weights to the conv layer, to  improve the performance
    var = var + scaleFacotr
    rstd = 1. / sqrt(var + eps)
    w = w * rstd * scale
    b = (b - mean) * rstd * scale + shift
    '''
    for key in train_net.params.iterkeys():
        if type(train_net.params[key]) is caffe._caffe.BlobVec:
            if key.endswith("/bn") or key.endswith("/scale"):
		continue
            else:
                conv = train_net.params[key]
                if not train_net.params.has_key(key + "/bn"):
                    for i, w in enumerate(conv):
                        deploy_net.params[key][i].data[...] = w.data
                else:
                    bn = train_net.params[key + "/bn"]
                    scale = train_net.params[key + "/scale"]
                    wt = conv[0].data
                    channels = wt.shape[0]
                    bias = np.zeros(wt.shape[0])
                    if len(conv) > 1:
                        bias = conv[1].data
                    mean = bn[0].data
                    var = bn[1].data
                    scalef = bn[2].data

                    scales = scale[0].data
                    shift = scale[1].data

                    if scalef != 0:
                        scalef = 1. / scalef
                    mean = mean * scalef
                    var = var * scalef
                    rstd = 1. / np.sqrt(var + 1e-5)
                    rstd1 = rstd.reshape((channels,1,1,1))
                    scales1 = scales.reshape((channels,1,1,1))
                    wt = wt * rstd1 * scales1
                    bias = (bias - mean) * rstd * scales + shift
                    
                    deploy_net.params[key][0].data[...] = wt
                    deploy_net.params[key][1].data[...] = bias
  

train_net = caffe.Net(train_proto, train_model, caffe.TRAIN)  
deploy_net = caffe.Net(deploy_proto, caffe.TEST)  

merge_bn(train_net, deploy_net)
deploy_net.save(save_model)

