from __future__ import print_function
import argparse
import argparse, time, logging, os, math
import datetime
from os import listdir
from os.path import isfile, join
import numpy as np
import mxnet as mx
import matplotlib.pyplot as plt
import csv
from mxnet import gluon, nd, image
from mxnet.gluon.data.vision import transforms
import gluoncv as gcv
gcv.utils.check_version('0.6.0')
from gluoncv.model_zoo import get_model
import gc
from mxnet import profiler
import MV3CNN as cnn
import sys

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

parser = argparse.ArgumentParser(description='Predict CIFAR10 classes from a given image')
parser.add_argument('--model', type=str, required=True,
                    help='name of the model to use')
parser.add_argument('--saved-params', type=str, default='',
                    help='path to the saved model parameters')
parser.add_argument('--input-pic', type=str, required=True,
                    help='path to the input picture')
parser.add_argument('--input-size', type=int, default=224,
                    help='size of the input image size. default is 224')
parser.add_argument('--crop-ratio', type=float, default=0.875,
                    help='Crop ratio during validation. default is 0.875')
parser.add_argument('--dtype', type=str, default='float32',
                    help='data type for training. default is float32')      
parser.add_argument('--use-pretrained', action='store_true',
                    help='enable using pretrained model from gluon.')
opt = parser.parse_args()

classes = 1000

context = [mx.cpu()]

# Load Model
model_name = opt.model
pretrained = True if opt.saved_params == '' else False
kwargs = {'classes': classes, 'pretrained': opt.use_pretrained}
net = cnn.mobilenet_v3_small(**kwargs)
#net = get_model(model_name, **kwargs)
net.cast(opt.dtype)
if not pretrained:
    net.load_parameters(opt.saved_params, ctx = context, cast_dtype=True)

profiler.set_config(profile_all=True,
                    aggregate_stats=True,
                    continuous_dump=True)


onlyfiles = [opt.input_pic+'/'+f for f in listdir(opt.input_pic) if isfile(join(opt.input_pic, f))]
#print (onlyfiles)

# Transform
normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
input_size = opt.input_size
crop_ratio = opt.crop_ratio if opt.crop_ratio > 0 else 0.875
resize = int(math.ceil(input_size / crop_ratio))
transform_fn = transforms.Compose([
    transforms.Resize(resize, keep_ratio=True),
    transforms.CenterCrop(input_size),
    transforms.ToTensor(),
    normalize
])

asd = 0
for iimg in onlyfiles:
    asd+=1
    # Load Images
    img = image.imread(iimg)
    img = transform_fn(img)
    #start_time = datetime.datetime.now()
    profiler.set_state('run')
    pred = net(img.expand_dims(0))
    mx.nd.waitall()
    profiler.set_state('stop')
    if asd%200==0:
        eprint(asd)

print(profiler.dumps(reset=False))