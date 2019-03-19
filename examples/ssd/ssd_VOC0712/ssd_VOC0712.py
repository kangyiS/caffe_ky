'''
Author: Mr.K

You can use this script to create a neural network and relative files.
I defined a file structure to make all files be clear.
The file structure is as followed:
                                                      |--data/ (store lmdb link, label map, and some scripts) 
                                                      |
                                                      |--log/ (store training log)
                                                      |
caffe/examples/(net name)/(net name and dataset)/-----|--model/ (store weights model including snapshot)
                                                      |
                                                      |--prototxt/ (store train.prototxt, test.prototxt, deploy.prototxt and solve.prototxt)
                                                      |
                                                      |--result/ (store detect results, test results and curve images)
                                                      |
                                                      |--files (some python and shell scripts)

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

network_name = "ssd"  #your neural net name, need to config
dataset_name = "VOC0712"#your dataset name, need to config
model_name = "{}_{}".format(network_name,dataset_name)
job_dir = "examples/{}/{}".format(network_name, model_name)
prototxt_dir = "{}/prototxt".format(job_dir)
trainLog_dir = "{}/log".format(job_dir)
trainData_dir = "{}/data".format(job_dir)
trainModel_dir = "{}/model".format(job_dir)

# The pretrained model. We use the Fully convolutional reduced (atrous) VGGNet.
pretrain_model = "{}/VGG_ILSVRC_16_layers_fc_reduced.caffemodel".format(trainModel_dir)#need to config

# some important files
train_net_file = "{}/train.prototxt".format(prototxt_dir) 
test_net_file = "{}/test.prototxt".format(prototxt_dir)
deploy_net_file = "{}/deploy.prototxt".format(prototxt_dir)
solver_file = "{}/solver.prototxt".format(prototxt_dir)
job_file = "{}/{}.sh".format(job_dir, model_name)


resume_training = True#need to config
# train ASAP
run_soon = 0

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
    'use_batchnorm': False,
    'mbox_source_layers': ['conv4_3', 'fc7', 'conv6_2', 'conv7_2', 'conv8_2', 'conv9_2'],
    'prior_variance': [0.1, 0.1, 0.2, 0.2],
    'steps': [8, 16, 32, 64, 100, 300],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'normalizations': [20, -1, -1, -1, -1, -1],
    'flip': True,
    'clip': False,
    'lr_mult': 1,
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
    'type': "SGD",#need to config
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

def AddExtraLayers(net, use_batchnorm=False, lr_mult=1, use_relu=True):
    # Add additional convolutional layers.
    # 18 x 18
    from_layer = net.keys()[-1]

    # 18 x 18
    out_layer = "conv6_1"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 256, 1, 0, 1,
        lr_mult=lr_mult)

    # 9 x 9
    from_layer = out_layer
    out_layer = "conv6_2"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 512, 3, 1, 2,
        lr_mult=lr_mult)

    # 9 x 9
    from_layer = out_layer
    out_layer = "conv7_1"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 128, 1, 0, 1,
      lr_mult=lr_mult)
    
    # 5 x 5
    from_layer = out_layer
    out_layer = "conv7_2"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 256, 3, 1, 2,
      lr_mult=lr_mult)

    # 5 x 5
    from_layer = out_layer
    out_layer = "conv8_1"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 128, 1, 0, 1,
      lr_mult=lr_mult)

    # 3 x 3
    from_layer = out_layer
    out_layer = "conv8_2"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 256, 3, 0, 1,
      lr_mult=lr_mult)

    # 3 x 3
    from_layer = out_layer
    out_layer = "conv9_1"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 128, 1, 0, 1,
      lr_mult=lr_mult)

    # 1 x 1
    from_layer = out_layer
    out_layer = "conv9_2"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 256, 3, 0, 1,
      lr_mult=lr_mult)

    return net

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

VGGNetBody(net, from_layer='data', fully_conv=True, reduced=True, dilated=True,
    dropout=False)

AddExtraLayers(net)

mbox_layers = CreateMultiBoxHead(net, data_layer='data', from_layers=mboxLayer_param['mbox_source_layers'],
        use_batchnorm=mboxLayer_param['use_batchnorm'], min_sizes=min_sizes, max_sizes=max_sizes,
        aspect_ratios=mboxLayer_param['aspect_ratios'], steps=mboxLayer_param['steps'], normalizations=mboxLayer_param['normalizations'],
        num_classes=mboxLayer_param['multibox_loss_param']['num_classes'], share_location=mboxLayer_param['multibox_loss_param']['share_location'],
        flip=mboxLayer_param['flip'], clip=mboxLayer_param['clip'],
        prior_variance=mboxLayer_param['prior_variance'], kernel_size=3, pad=1, lr_mult=mboxLayer_param['lr_mult'])
# Create the MultiBoxLossLayer, for train net.
mbox_layers.append(net.label)
net['mbox_loss'] = L.MultiBoxLoss(*mbox_layers, multibox_loss_param=mboxLayer_param['multibox_loss_param'],
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

VGGNetBody(net, from_layer='data', fully_conv=True, reduced=True, dilated=True,
    dropout=False)

AddExtraLayers(net)

mbox_layers = CreateMultiBoxHead(net, data_layer='data', from_layers=mboxLayer_param['mbox_source_layers'],
        use_batchnorm=mboxLayer_param['use_batchnorm'], min_sizes=min_sizes, max_sizes=max_sizes,
        aspect_ratios=mboxLayer_param['aspect_ratios'], steps=mboxLayer_param['steps'], normalizations=mboxLayer_param['normalizations'],
        num_classes=mboxLayer_param['multibox_loss_param']['num_classes'], share_location=mboxLayer_param['multibox_loss_param']['share_location'], 
        flip=mboxLayer_param['flip'], clip=mboxLayer_param['clip'],
        prior_variance=mboxLayer_param['prior_variance'], kernel_size=3, pad=1, lr_mult=mboxLayer_param['lr_mult'])

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
deploy_net = net
with open(deploy_net_file, 'w') as f:
    net_param = deploy_net.to_proto()
    # Remove the first (AnnotatedData) and last (DetectionEvaluate) layer from test net.
    del net_param.layer[0]
    del net_param.layer[-1]
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

