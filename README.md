# Faster MobileNets
This repo contains implementation of several architectures based on MobileNet NN family, scripts to train them, measure inference time and data measurements conducted on RaspberryPi4B. 

All models are in separate folder in /models. Each folder also contains training script based on Gluon code which you can find [here](https://gluon-cv.mxnet.io/build/examples_classification/dive_deep_imagenet.html) (in case architecture uses different optimizer etc.) and testing script which measure inference time on batches from ImageNet validation dataset using MxNet built in profiler

Using runMeasurment.sh you can run all batches 3 times and save scores to separate files. 

In “Profiler output formatting” folder you can find simple script that combines scores from all runs into several files, each containing computation time of specific layer type (you have to match version of formatter with architecture you benchmarking)

Measurements folder contains results of benchmarks mentioned at the beginning.
