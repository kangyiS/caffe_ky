'''
Author: Mr.K

This file is used to train MobileNet-SSD network.
After training, you are supposed to merge batchNorm layers to get a deploy model.
The file used to merge batchNorm layers is in the current directory 
'''

from __future__ import print_function
import caffe
from caffe.model_libs import *
from google.protobuf import text_format

import math
import os
import shutil
import stat
import subprocess
import sys
####### you are supposed to run this script at the CAFFE ROOT ########################

network_name = "MobileSSD"  #your neural net name, need to config
dataset_name = "VOC0712"#your dataset name, need to config
model_name = "{}py_{}".format(network_name,dataset_name)
job_dir = "examples/{}/{}".format(network_name, model_name)
prototxt_dir = "{}/prototxt".format(job_dir)
trainLog_dir = "{}/log".format(job_dir)
trainData_dir = "{}/data".format(job_dir)
trainModel_dir = "{}/model".format(job_dir)

# The pretrained model. We use the Fully convolutional reduced (atrous) VGGNet.
pretrain_model = "{}/mobilenet_iter_73000.caffemodel".format(trainModel_dir)#need to config

# some important files
train_net_file = "{}/train.prototxt".format(prototxt_dir) 
test_net_file = "{}/test.prototxt".format(prototxt_dir)
deploy_net_file = "{}/deploy.prototxt".format(prototxt_dir)
solver_file = "{}/solver.prototxt".format(prototxt_dir)
job_file = "{}/{}.sh".format(job_dir, model_name)


resume_training = True#need to config
# train ASAP
run_soon = 0#True

dataLayer_param = {
    'train_data': "{}/VOC0712_trainval_lmdb".format(trainData_dir),#need to config
    'test_data': "{}/VOC0712_test_lmdb".format(trainData_dir),#need to config
    'num_test_image': 4952,#need to config
    'train_batch_size': 32,#need to config
    'test_batch_size': 8,#need to config
    'label_map_file': "{}/labelmap_voc.prototxt".format(trainData_dir),#need to config
    'train_transform_param': 
    {
        'mirror': True,
        'mean_value': [104, 117, 123],
        'resize_param': 
        {
            'prob': 1,
            'resize_mode': P.Resize.WARP,
            # the size of input images, for training
            'height': 300, #need to config
            'width': 300, #need to config
            'interp_mode': [
                P.Resize.LINEAR,
                P.Resize.AREA,
                P.Resize.NEAREST,
                P.Resize.CUBIC,
                P.Resize.LANCZOS4,
                ],
        },
        'distort_param': 
        {
            'brightness_prob': 0.5,
            'brightness_delta': 32,
            'contrast_prob': 0.5,
            'contrast_lower': 0.5,
            'contrast_upper': 1.5,
            'hue_prob': 0.5,
            'hue_delta': 18,
            'saturation_prob': 0.5,
            'saturation_lower': 0.5,
            'saturation_upper': 1.5,
            'random_order_prob': 0.0,
        },
        'expand_param': 
        {
            'prob': 0.5,
            'max_expand_ratio': 4.0,
        },
        'emit_constraint': 
        {
            'emit_type': caffe_pb2.EmitConstraint.CENTER,
        }
    },
    'test_transform_param': 
    {
        'mean_value': [104, 117, 123],
        'resize_param': 
        {
            'prob': 1,
            'resize_mode': P.Resize.WARP,
            # the size of input images, for testing
            'height': 300,#need to config
            'width': 300,#need to config
            'interp_mode': [P.Resize.LINEAR],
        }
    },
    'batch_sampler': 
    [{
        'sampler': {},
        'max_trials': 1,
        'max_sample': 1,
    },
    {
        'sampler': 
        {
            'min_scale': 0.3,
            'max_scale': 1.0,
            'min_aspect_ratio': 0.5,
            'max_aspect_ratio': 2.0,
        },
        'sample_constraint': 
        {
            'min_jaccard_overlap': 0.1,
        },
            'max_trials': 50,
            'max_sample': 1,
    },
    {
        'sampler': 
        {
            'min_scale': 0.3,
            'max_scale': 1.0,
            'min_aspect_ratio': 0.5,
            'max_aspect_ratio': 2.0,
        },
        'sample_constraint': 
        {
            'min_jaccard_overlap': 0.3,
        },
        'max_trials': 50,
        'max_sample': 1,
    },
    {
        'sampler': 
        {
            'min_scale': 0.3,
            'max_scale': 1.0,
            'min_aspect_ratio': 0.5,
            'max_aspect_ratio': 2.0,
        },
        'sample_constraint': 
        {
            'min_jaccard_overlap': 0.5,
        },
        'max_trials': 50,
        'max_sample': 1,
    },
    {
        'sampler': 
        {
            'min_scale': 0.3,
            'max_scale': 1.0,
            'min_aspect_ratio': 0.5,
            'max_aspect_ratio': 2.0,
        },
        'sample_constraint': 
        {
            'min_jaccard_overlap': 0.7,
        },
        'max_trials': 50,
        'max_sample': 1,
    },
    {
        'sampler': 
        {
            'min_scale': 0.3,
            'max_scale': 1.0,
            'min_aspect_ratio': 0.5,
            'max_aspect_ratio': 2.0,
        },
        'sample_constraint': 
        {
            'min_jaccard_overlap': 0.9,
        },
        'max_trials': 50,
        'max_sample': 1,
    },
    {
        'sampler': 
        {
            'min_scale': 0.3,
            'max_scale': 1.0,
            'min_aspect_ratio': 0.5,
            'max_aspect_ratio': 2.0,
        },
        'sample_constraint': 
        {
            'max_jaccard_overlap': 1.0,
        },
        'max_trials': 50,
        'max_sample': 1,
    }]
}

mboxLayer_param = {
    'mbox_source_layers': ['conv11', 'conv13', 'conv14_2', 'conv15_2', 'conv16_2', 'conv17_2'],
    'prior_variance': [0.1, 0.1, 0.2, 0.2],
    'steps': [8, 16, 32, 64, 100, 300],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'flip': True,
    'clip': False,
    'multibox_loss_param': 
    {
        'loc_loss_type': P.MultiBoxLoss.SMOOTH_L1,
        'conf_loss_type': P.MultiBoxLoss.SOFTMAX,
        'loc_weight': 1.0,  #loc_weight = (neg_pos_ratio + 1.) / 4.
        'num_classes': 21,#need to config
        'share_location': True,
        'match_type': P.MultiBoxLoss.PER_PREDICTION,
        'overlap_threshold': 0.5,
        'use_prior_for_matching': True,
        'background_label_id': 0,
        'use_difficult_gt': True,
        'mining_type': P.MultiBoxLoss.MAX_NEGATIVE,
        'neg_pos_ratio': 3,
        'neg_overlap': 0.5,
        'code_type': P.PriorBox.CENTER_SIZE,
        'ignore_cross_boundary_bbox': False,
    },
    'loss_param': 
    {
        'normalization': P.Loss.VALID,
    }
}

# parameters for generating detection output.
det_out_param = {
    'num_classes': mboxLayer_param['multibox_loss_param']['num_classes'],
    'share_location': mboxLayer_param['multibox_loss_param']['share_location'],
    'background_label_id': 0,
    'nms_param': {'nms_threshold': 0.45, 'top_k': 400},
    'save_output_param': {
        'output_directory': "{}/result".format(job_dir),
        'output_name_prefix': "comp4_det_test_",
        'output_format': "VOC",
        'label_map_file': dataLayer_param['label_map_file'],
        # Stores the test image names and sizes. Created by examples/ssd/ssd_VOC0712/data/create_lmdb.py
        'name_size_file': "{}/test_name_size.txt".format(trainData_dir),
        'num_test_image': dataLayer_param['num_test_image'],
        },
    'keep_top_k': 200,
    'confidence_threshold': 0.01,
    'code_type': mboxLayer_param['multibox_loss_param']['code_type'],
}

# parameters for evaluating detection results.
det_eval_param = {
    'num_classes': mboxLayer_param['multibox_loss_param']['num_classes'],
    'background_label_id': 0,
    'overlap_threshold': 0.5,
    'evaluate_difficult_gt': False,
    'name_size_file': det_out_param['save_output_param']['name_size_file'],
}

# Ideally test_batch_size should be divisible by num_test_image,
# otherwise mAP will be slightly off the true value.
test_iter = int(math.ceil(float(dataLayer_param['num_test_image']) / dataLayer_param['test_batch_size']))

solver_param = {
    # base learning rate
    'base_lr': 0.001,#need to config
    'weight_decay': 0.0005,
    'lr_policy': "multistep",#need to config
    'stepvalue': [80000, 100000, 120000],#need to config
    'gamma': 0.1,
    'momentum': 0.9,
    'iter_size': 32/dataLayer_param['train_batch_size'],
    'max_iter': 120000,#need to config
    'snapshot': 10000,#need to config
    'display': 10,
    'average_loss': 10,
    'type': "RMSProp",#need to config
    'solver_mode': P.Solver.GPU,
    'device_id': 0,
    'debug_info': False,
    'snapshot_after_train': True,
    'test_iter': [test_iter],
    'test_interval': 200,#need to config
    'eval_type': "detection",
    'ap_version': "11point",
    'test_initialization': False,
}

min_ratio = 20 # in percent %
max_ratio = 90 # in percent %
min_dim = 300
step = int(math.floor((max_ratio - min_ratio) / (len(mboxLayer_param['mbox_source_layers']) - 2)))
min_sizes = []
max_sizes = []
for ratio in xrange(min_ratio, max_ratio + 1, step):
    min_sizes.append(min_dim * ratio / 100.)
    max_sizes.append(min_dim * (ratio + step) / 100.)
min_sizes = [min_dim * 10 / 100.] + min_sizes
max_sizes = [min_dim * 20 / 100.] + max_sizes

def Conv_BN_Sc_ReLU_Layer(net, from_layer, out_layer, num_output, pad, stride, kernel_size,  
                          lr_mult=1, group=1, engine=0, net_stage="train", conv_type="conv"):
    if net_stage == "train" or net_stage == "test":
        net[out_layer] = L.Convolution(net[from_layer],num_output=num_output,pad=pad,stride=stride,
                           kernel_size=kernel_size,group=group,engine=engine,bias_term=False,
                           weight_filler=dict(type="msra"),param=dict(lr_mult=lr_mult,decay_mult=lr_mult))
    
        layer_name = "{}/bn".format(out_layer)
        net[layer_name] = L.BatchNorm(net[out_layer],in_place=True,
                            param=[dict(lr_mult=0,decay_mult=0),dict(lr_mult=0,decay_mult=0),dict(lr_mult=0,decay_mult=0)])

        layer_name = "{}/scale".format(out_layer)
        net[layer_name] = L.Scale(net[out_layer],param=[dict(lr_mult=lr_mult,decay_mult=0),dict(lr_mult=lr_mult*2,decay_mult=0)],
                            filler=dict(value=1),bias_term=True,bias_filler=dict(value=0),in_place=True)

        layer_name = "{}/relu".format(out_layer)
        net[layer_name] = L.ReLU(net[out_layer],in_place=True)
    elif net_stage == "deploy":
        if conv_type == "conv":
            net[out_layer] = L.Convolution(net[from_layer],num_output=num_output,pad=pad,stride=stride,
                               kernel_size=kernel_size,group=group,
                               weight_filler=dict(type="msra"),bias_filler=dict(type="constant",value=0),
                               param=[dict(lr_mult=lr_mult,decay_mult=lr_mult),dict(lr_mult=lr_mult*2,decay_mult=0)])
        elif conv_type == "dw_conv":
            net[out_layer] = L.DepthwiseConvolution(net[from_layer],num_output=num_output,pad=pad,stride=stride,
                               kernel_size=kernel_size,group=group,
                               weight_filler=dict(type="msra"),bias_filler=dict(type="constant",value=0),
                               param=[dict(lr_mult=lr_mult,decay_mult=lr_mult),dict(lr_mult=lr_mult*2,decay_mult=0)])
        else:
            print("Incorrect conv type !")
        layer_name = "{}/relu".format(out_layer)
        net[layer_name] = L.ReLU(net[out_layer],in_place=True)
    else:
        print("Incorrect net stage !")

def MobileSSD_Body(net, from_layer, lr_mult, net_stage):
#                                 in         out   num  p  s  k  lr_mult  g  e  net_stage   conv_type
    Conv_BN_Sc_ReLU_Layer(net, from_layer, "conv0", 32, 1, 2, 3, lr_mult, 1, 0, net_stage)
    Conv_BN_Sc_ReLU_Layer(net, "conv0", "conv1/dw", 32, 1, 1, 3, lr_mult, 32, 1, net_stage)
    Conv_BN_Sc_ReLU_Layer(net, "conv1/dw", "conv1", 64, 0, 1, 1, lr_mult, 1, 0, net_stage)
    Conv_BN_Sc_ReLU_Layer(net, "conv1", "conv2/dw", 64, 1, 2, 3, lr_mult, 64, 1, net_stage)
    Conv_BN_Sc_ReLU_Layer(net, "conv2/dw", "conv2", 128, 0, 1, 1, lr_mult, 1, 0, net_stage)
    Conv_BN_Sc_ReLU_Layer(net, "conv2", "conv3/dw", 128, 1, 1, 3, lr_mult, 128, 1, net_stage)
    Conv_BN_Sc_ReLU_Layer(net, "conv3/dw", "conv3", 128, 0, 1, 1, lr_mult, 1, 0, net_stage)
    Conv_BN_Sc_ReLU_Layer(net, "conv3", "conv4/dw", 128, 1, 2, 3, lr_mult, 128, 1, net_stage)
    Conv_BN_Sc_ReLU_Layer(net, "conv4/dw", "conv4", 256, 0, 1, 1, lr_mult, 1, 0, net_stage)
    Conv_BN_Sc_ReLU_Layer(net, "conv4", "conv5/dw", 256, 1, 1, 3, lr_mult, 256, 1, net_stage)
    Conv_BN_Sc_ReLU_Layer(net, "conv5/dw", "conv5", 256, 0, 1, 1, lr_mult, 1, 0, net_stage)
    Conv_BN_Sc_ReLU_Layer(net, "conv5", "conv6/dw", 256, 1, 2, 3, lr_mult, 256, 1, net_stage)
    Conv_BN_Sc_ReLU_Layer(net, "conv6/dw", "conv6", 512, 0, 1, 1, lr_mult, 1, 0, net_stage)
    Conv_BN_Sc_ReLU_Layer(net, "conv6", "conv7/dw", 512, 1, 1, 3, lr_mult, 512, 1, net_stage)
    Conv_BN_Sc_ReLU_Layer(net, "conv7/dw", "conv7", 512, 0, 1, 1, lr_mult, 1, 0, net_stage)
    Conv_BN_Sc_ReLU_Layer(net, "conv7", "conv8/dw", 512, 1, 1, 3, lr_mult, 512, 1, net_stage)
    Conv_BN_Sc_ReLU_Layer(net, "conv8/dw", "conv8", 512, 0, 1, 1, lr_mult, 1, 0, net_stage)
    Conv_BN_Sc_ReLU_Layer(net, "conv8", "conv9/dw", 512, 1, 1, 3, lr_mult, 512, 1, net_stage)
    Conv_BN_Sc_ReLU_Layer(net, "conv9/dw", "conv9", 512, 0, 1, 1, lr_mult, 1, 0, net_stage)
    Conv_BN_Sc_ReLU_Layer(net, "conv9", "conv10/dw", 512, 1, 1, 3, lr_mult, 512, 1, net_stage)
    Conv_BN_Sc_ReLU_Layer(net, "conv10/dw", "conv10", 512, 0, 1, 1, lr_mult, 1, 0, net_stage)
    Conv_BN_Sc_ReLU_Layer(net, "conv10", "conv11/dw", 512, 1, 1, 3, lr_mult, 512, 1, net_stage)
    Conv_BN_Sc_ReLU_Layer(net, "conv11/dw", "conv11", 512, 0, 1, 1, lr_mult, 1, 0, net_stage)
    Conv_BN_Sc_ReLU_Layer(net, "conv11", "conv12/dw", 512, 1, 2, 3, lr_mult, 512, 1, net_stage)
    Conv_BN_Sc_ReLU_Layer(net, "conv12/dw", "conv12", 1024, 0, 1, 1, lr_mult, 1, 0, net_stage)
    Conv_BN_Sc_ReLU_Layer(net, "conv12", "conv13/dw", 1024, 1, 1, 3, lr_mult, 1024, 1, net_stage)
    Conv_BN_Sc_ReLU_Layer(net, "conv13/dw", "conv13", 1024, 0, 1, 1, lr_mult, 1, 0, net_stage)
    Conv_BN_Sc_ReLU_Layer(net, "conv13", "conv14_1", 256, 0, 1, 1, lr_mult, 1, 0, net_stage)
    Conv_BN_Sc_ReLU_Layer(net, "conv14_1", "conv14_2", 512, 1, 2, 3, lr_mult, 1, 0, net_stage)
    Conv_BN_Sc_ReLU_Layer(net, "conv14_2", "conv15_1", 128, 0, 1, 1, lr_mult, 1, 0, net_stage)
    Conv_BN_Sc_ReLU_Layer(net, "conv15_1", "conv15_2", 256, 1, 2, 3, lr_mult, 1, 0, net_stage)
    Conv_BN_Sc_ReLU_Layer(net, "conv15_2", "conv16_1", 128, 0, 1, 1, lr_mult, 1, 0, net_stage)
    Conv_BN_Sc_ReLU_Layer(net, "conv16_1", "conv16_2", 256, 1, 2, 3, lr_mult, 1, 0, net_stage)
    Conv_BN_Sc_ReLU_Layer(net, "conv16_2", "conv17_1", 64, 0, 1, 1, lr_mult, 1, 0, net_stage)
    Conv_BN_Sc_ReLU_Layer(net, "conv17_1", "conv17_2", 128, 1, 2, 3, lr_mult, 1, 0, net_stage)

def MultiBoxHead(net, class_num, from_layers=[],max_sizes=[],min_sizes=[],aspect_ratios=[],steps=[], lr_mult=0.1, 
                 data_layer="data", img_height=0, img_width=0, flip=True, clip=True, offset=0.5, prior_variance = [0.1]):
    layer_num = len(from_layers)
    loc_layers = []
    conf_layers = []
    priorbox_layers = []
    for i in range(layer_num):
        min_size = min_sizes[i]
        if type(min_size) is not list:
            min_size = [min_size]
        if len(aspect_ratios) > i:
            aspect_ratio = aspect_ratios[i]
            if type(aspect_ratio) is not list:
                aspect_ratio = [aspect_ratio]
        max_size = []
        if len(max_sizes) > i:
            max_size = max_sizes[i]
            if type(max_size) is not list:
                max_size = [max_size]
            if max_size:
                assert len(max_size) == len(min_size), "max_size and min_size should have same length."
        if max_size:
            num_priors_per_location = (2 + len(aspect_ratio)) * len(min_size)
        else:
            num_priors_per_location = (1 + len(aspect_ratio)) * len(min_size)
        if flip:
            num_priors_per_location += len(aspect_ratio) * len(min_size)
        step = []
        if len(steps) > i:
            step = steps[i]

        # Create location prediction layer        
        loc_name = "{}_mbox_loc".format(from_layers[i])
        num_loc_output = num_priors_per_location * 4
        net[loc_name] = L.Convolution(net[from_layers[i]], num_output=num_loc_output, pad=0,stride=1,kernel_size=1,
                          weight_filler=dict(type="msra"),bias_filler=dict(type="constant",value=0.0),
                          param=[dict(lr_mult=lr_mult,decay_mult=lr_mult),dict(lr_mult=lr_mult*2,decay_mult=0.0)])
        perm_name = "{}_mbox_loc_perm".format(from_layers[i])
        net[perm_name] = L.Permute(net[loc_name],order=[0,2,3,1])
        flat_name = "{}_mbox_loc_flat".format(from_layers[i])
        net[flat_name] = L.Flatten(net[perm_name],axis=1)
        loc_layers.append(net[flat_name])

        # Create confidence prediction layer
        conf_name = "{}_mbox_conf".format(from_layers[i])
        num_conf_output = num_priors_per_location * class_num
        net[conf_name] = L.Convolution(net[from_layers[i]],num_output=num_conf_output,pad=0,stride=1,kernel_size=1,
                           weight_filler=dict(type="msra"),bias_filler=dict(type="constant",value=0.0),
                           param=[dict(lr_mult=1.0,decay_mult=1.0),dict(lr_mult=2.0,decay_mult=0.0)])
        perm_name = "{}_mbox_conf_perm".format(from_layers[i])
        net[perm_name] = L.Permute(net[conf_name],order=[0,2,3,1])
        flat_name = "{}_mbox_conf_flat".format(from_layers[i])
        net[flat_name] = L.Flatten(net[perm_name],axis=1)
        conf_layers.append(net[flat_name])
        
        # Creat prior generation layer
        prior_name = "{}_mbox_priorbox".format(from_layers[i])
        net[prior_name] = L.PriorBox(net[from_layers[i]], net[data_layer], min_size=min_size,
                            clip=clip, variance=prior_variance, offset=offset)
        if max_size:
            net.update(prior_name, {'max_size': max_size})
        if aspect_ratio:
            net.update(prior_name, {'aspect_ratio': aspect_ratio, 'flip': flip})
        if step:
            net.update(prior_name, {'step': step})
        if img_height != 0 and img_width != 0:
            if img_height == img_width:
                net.update(prior_name, {'img_size': img_height})
            else:
                net.update(prior_name, {'img_h': img_height, 'img_w': img_width})
        priorbox_layers.append(net[prior_name])
    mbox_layers = []
    name = "mbox_loc"
    net[name] = L.Concat(*loc_layers, axis=1)
    mbox_layers.append(net[name])
    name = "mbox_conf"
    net[name] = L.Concat(*conf_layers, axis=1)
    mbox_layers.append(net[name])
    name = "mbox_priorbox"
    net[name] = L.Concat(*priorbox_layers, axis=2)
    mbox_layers.append(net[name])
    return mbox_layers

# Check file.
check_if_exist(dataLayer_param['train_data'])
check_if_exist(dataLayer_param['test_data'])
check_if_exist(dataLayer_param['label_map_file'])
check_if_exist(pretrain_model)
make_if_not_exist(prototxt_dir)
make_if_not_exist(trainLog_dir)
make_if_not_exist(trainData_dir)
make_if_not_exist(trainModel_dir)
# Create train net.
net = caffe.NetSpec()
net.data, net.label = CreateAnnotatedDataLayer(dataLayer_param['train_data'], batch_size=dataLayer_param['train_batch_size'],
        train=True, output_label=True, label_map_file=dataLayer_param['label_map_file'],
        transform_param=dataLayer_param['train_transform_param'], batch_sampler=dataLayer_param['batch_sampler'])

MobileSSD_Body(net, "data", 0.1, "train")

mbox_layers = MultiBoxHead(net, from_layers=mboxLayer_param['mbox_source_layers'],min_sizes=min_sizes, max_sizes=max_sizes, 
                           steps=mboxLayer_param['steps'], class_num=mboxLayer_param['multibox_loss_param']['num_classes'],
                           aspect_ratios=mboxLayer_param['aspect_ratios'],  prior_variance=mboxLayer_param['prior_variance'],
                           flip=mboxLayer_param['flip'], clip=mboxLayer_param['clip'], lr_mult=0.1)

# Create the MultiBoxLossLayer, for train net.
mbox_layers.append(net.label)
net['mbox_loss'] = L.MultiBoxLossss(*mbox_layers, multibox_loss_param=mboxLayer_param['multibox_loss_param'],
        loss_param=mboxLayer_param['loss_param'], include=dict(phase=caffe_pb2.Phase.Value('TRAIN')),
        propagate_down=[True, True, False, False])

# Create the train net file (train.prototxt)
with open(train_net_file, 'w') as f:
    print('name: "{}_train"'.format(model_name), file=f)
    print(net.to_proto(), file=f)

# Create test net.
net = caffe.NetSpec()
net.data, net.label = CreateAnnotatedDataLayer(dataLayer_param['test_data'], batch_size=dataLayer_param['test_batch_size'],
        train=False, output_label=True, label_map_file=dataLayer_param['label_map_file'],
        transform_param=dataLayer_param['test_transform_param'])

MobileSSD_Body(net, "data", 1, "test")

mbox_layers = MultiBoxHead(net, from_layers=mboxLayer_param['mbox_source_layers'],min_sizes=min_sizes, max_sizes=max_sizes, 
                           steps=mboxLayer_param['steps'], class_num=mboxLayer_param['multibox_loss_param']['num_classes'],
                           aspect_ratios=mboxLayer_param['aspect_ratios'],  prior_variance=mboxLayer_param['prior_variance'],
                           flip=mboxLayer_param['flip'], clip=mboxLayer_param['clip'], lr_mult=1)

net['mbox_conf_reshape'] = L.Reshape(net['mbox_conf'], shape=dict(dim=[0, -1, mboxLayer_param['multibox_loss_param']['num_classes']]))
net['mbox_conf_softmax'] = L.Softmax(net['mbox_conf_reshape'], axis=2)
net['mbox_conf_flatten'] = L.Flatten(net['mbox_conf_softmax'], axis=1)
#0,1,2 for mbox_loc,mbox_conf,mbox_priorbox
mbox_layers[1] = net['mbox_conf_flatten']

net['detection_out'] = L.DetectionOutput(*mbox_layers, detection_output_param=det_out_param, include=dict(phase=caffe_pb2.Phase.Value('TEST')))
net['detection_eval'] = L.DetectionEvaluate(net['detection_out'], net['label'], detection_evaluate_param=det_eval_param, include=dict(phase=caffe_pb2.Phase.Value('TEST')))

# Create the test net file (test.prototxt)
with open(test_net_file, 'w') as f:
    print('name: "{}_test"'.format(model_name), file=f)
    print(net.to_proto(), file=f)

# Create the deploy file (deploy.prototxt)
# Remove the first and last layer from test net.
#deploy_net = net
net = caffe.NetSpec()

#net["data"] = L.Data(net)
net.data, net.label = CreateAnnotatedDataLayer(dataLayer_param['test_data'], batch_size=dataLayer_param['test_batch_size'],
        train=False, output_label=True, label_map_file=dataLayer_param['label_map_file'],
        transform_param=dataLayer_param['test_transform_param'])
MobileSSD_Body(net, "data", 1, "deploy")

mbox_layers = MultiBoxHead(net, from_layers=mboxLayer_param['mbox_source_layers'],min_sizes=min_sizes, max_sizes=max_sizes, 
                           steps=mboxLayer_param['steps'], class_num=mboxLayer_param['multibox_loss_param']['num_classes'],
                           aspect_ratios=mboxLayer_param['aspect_ratios'],  prior_variance=mboxLayer_param['prior_variance'],
                           flip=mboxLayer_param['flip'], clip=mboxLayer_param['clip'], lr_mult=1)

net['mbox_conf_reshape'] = L.Reshape(net['mbox_conf'], shape=dict(dim=[0, -1, mboxLayer_param['multibox_loss_param']['num_classes']]))
net['mbox_conf_softmax'] = L.Softmax(net['mbox_conf_reshape'], axis=2)
net['mbox_conf_flatten'] = L.Flatten(net['mbox_conf_softmax'], axis=1)
#0,1,2 for mbox_loc,mbox_conf,mbox_priorbox
mbox_layers[1] = net['mbox_conf_flatten']

net['detection_out'] = L.DetectionOutput(*mbox_layers, detection_output_param=det_out_param, include=dict(phase=caffe_pb2.Phase.Value('TEST')))

with open(deploy_net_file, 'w') as f:
    net_param = net.to_proto()
    del net_param.layer[0]
    net_param.name = '{}_deploy'.format(model_name)
    net_param.input.extend(['data'])
    net_param.input_shape.extend([
        caffe_pb2.BlobShape(dim=[1, 3, dataLayer_param['test_transform_param']['resize_param']['height'], dataLayer_param['test_transform_param']['resize_param']['width']])])
    print(net_param, file=f)


# Create solver.
solver = caffe_pb2.SolverParameter(
        train_net = train_net_file,
        test_net = [test_net_file],
        snapshot_prefix = "{}/{}".format(trainModel_dir,model_name),
        **solver_param)

# Create the solver file (solver.prototxt)
with open(solver_file, 'w') as f:
    print(solver, file=f)

max_iter = 0
# Find most recent snapshot.
for file in os.listdir(trainModel_dir):
    if file.endswith(".solverstate"):
        basename = os.path.splitext(file)[0]
        iter = int(basename.split("{}_iter_".format(model_name))[1])
        if iter > max_iter:
            max_iter = iter

train_src_param = '--weights="{}" \\\n'.format(pretrain_model)
if resume_training:
    if max_iter > 0:
        train_src_param = '--snapshot="{}/{}_iter_{}.solverstate" \\\n'.format(trainModel_dir, model_name, max_iter)

# We assume you are running the script at the CAFFE ROOT.
caffe_root = os.getcwd()

# Create job file.
with open(job_file, 'w') as f:
    f.write('cd {}\n'.format(caffe_root))
    f.write('./build/tools/caffe train \\\n')
    f.write('--solver="{}" \\\n'.format(solver_file))
    f.write(train_src_param)
    if solver_param['solver_mode'] == P.Solver.GPU:
        f.write('--gpu 0 2>&1 | tee {}/{}.log\n'.format(trainLog_dir, model_name))
    else:
        f.write('2>&1 | tee {}/{}.log\n'.format(trainLog_dir, model_name))
# Run the job.
os.chmod(job_file, stat.S_IRWXU)
if run_soon:
    subprocess.call(job_file, shell=True)

