#!/usr/bin/env python
# Master's Thesis - Depth Estimation by Convolutional Neural Networks
# Jan Ivanecky; xivane00@stud.fit.vutbr.cz

from __future__ import print_function	
import numpy as np
import matplotlib.pyplot as plt
import sys
from PIL import Image
import cv2
#import cv
import os.path
os.environ['GLOG_minloglevel'] = '2' 
import caffe
import scipy.ndimage
import argparse
import operator	
import shutil

WIDTH = 298
HEIGHT = 218
OUT_WIDTH = 74
OUT_HEIGHT = 54

def ProcessToOutput(depth):
    depth = np.clip(depth, 0.001, 1000)	
    return np.clip(2 * 0.179581 * np.log(depth) + 1, 0, 1)

def testNet(net, img):	
    net.blobs['X'].data[...] = img	
    net.forward()
    output = net.blobs['depth-refine'].data
    return output

# models
caffemodel = "model/model_norm_abs_100k.caffemodel"
deployfile = "model/model_norm_abs_100k.prototxt"
caffe.set_mode_gpu()
net = caffe.Net(deployfile, caffemodel, caffe.TEST)
# camera
cap = cv2.VideoCapture(1)
print("VideoIsOpened:",cap.isOpened())
cap.set(3,WIDTH)
cap.set(4,HEIGHT)
# main loop
while(True):   
    ret, frame = cap.read()    
    cv2.imshow('RGB', frame)
    print("--------")
    print("input shape:",frame.shape)#input shape: (240, 320, 3)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    img = frame[:,:,::-1] # image inversion for image and camera
    img = caffe.io.resize(img, (HEIGHT, WIDTH, 3))  #(218, 298, 3)
    img = np.transpose(img, (2,0,1))                #(3, 218, 298)
    input = np.reshape(img, (1,3,HEIGHT,WIDTH))    
    input *= 255
    input -= 127

    print("input_net shape:",input.shape)#input_net shape: (1, 3, 218, 298)
    output = testNet(net, input) 
    print("output_net shape:",output.shape)#output_net shape: (1, 1, 54, 74)
    
    #rescale for output
    scaleW = float(WIDTH) / float(OUT_WIDTH)
    scaleH = float(HEIGHT) / float(OUT_HEIGHT)
                                        #output_net rescale: (1, 1, 218, 298)    
    output = scipy.ndimage.zoom(output, (1,1,scaleH,scaleW), order=3)
    output = ProcessToOutput(output)

    imgnp = np.reshape(output, (HEIGHT, WIDTH, 1))#output shape: (218, 298, 1)
    imgnp = np.array(imgnp * 255, dtype = np.uint8)
    cv2.imshow('depth',imgnp)

cap.release()    
cv2.destroyAllWindows()

