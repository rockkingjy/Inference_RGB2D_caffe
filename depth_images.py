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

# caffe model
caffemodel = "../../models/depth_model/model_norm_abs_100k.caffemodel"
deployfile = "../../models/depth_model/model_norm_abs_100k.prototxt"

# images
input_dir = "/media/elab/sdd/caffe/examples/cpp_depth"#"/media/enroutelab/sdd/mycodes/rtabmap/bin/0/rgb_sync"
output_dir = "/media/elab/sdd/caffe/examples/cpp_depth"#"/media/enroutelab/sdd/mycodes/rtabmap/bin/0/depth_sync"

WIDTH = 298 #width of input of neural net
HEIGHT = 218
OUT_WIDTH = 74 #width of output of neural net
OUT_HEIGHT = 54
SAVE_WIDTH = 640 #1920 #width of image for save
SAVE_HEIGHT = 480 #1080

def loadImage(path, channels, width, height):
	img = caffe.io.load_image(path)
	img = caffe.io.resize(img, (height, width, channels))
	img = np.transpose(img, (2,0,1))
	img = np.reshape(img, (1,channels,height,width))
	return img

def printImage(img, name, channels, width, height):
	params = list()
	#params.append(cv.CV_IMWRITE_PNG_COMPRESSION)
	#params.append(8)

	imgnp = np.reshape(img, (height,width, channels))
	imgnp = np.array(imgnp, dtype=np.float32) 
	#imgnp = np.array(imgnp * 255, dtype = np.uint8)
	cv2.imwrite(name, imgnp, params)

def ProcessToOutput(depth):
    depth = np.clip(depth, 0.001, 1000)	
    return np.clip(2 * 0.179581 * np.log(depth) + 1, 0, 1)

def testNet(net, img):	
    net.blobs['X'].data[...] = img	
    net.forward()
    output = net.blobs['depth-refine'].data
    return output

# models
caffe.set_mode_gpu()
net = caffe.Net(deployfile, caffemodel, caffe.TEST)

fileCount = len([name for name in os.listdir(input_dir)])


# get the depth image for one picture=====================
inputFileName = "rgb3.jpeg"
inputFilePath = input_dir + '/' + inputFileName
input = loadImage(inputFilePath, 3, WIDTH, HEIGHT)	
input *= 255
input -= 127

print("input_net shape:",input.shape)#input_net shape: (1, 3, 218, 298)
output = testNet(net, input) 
print("output_net shape:",output.shape)#output_net shape: (1, 1, 54, 74)
    
#rescale for output
scaleW = float(SAVE_WIDTH) / float(OUT_WIDTH)
scaleH = float(SAVE_HEIGHT) / float(OUT_HEIGHT)
                                        #output_net rescale: (1, 1, 218, 298)    
output = scipy.ndimage.zoom(output, (1,1,scaleH,scaleW), order=3)
output = ProcessToOutput(output)
print("Scaled_output_net shape:",output.shape)


imgnp = np.reshape(output, (SAVE_HEIGHT, SAVE_WIDTH, 1))#output shape: (218, 298, 1)
imgnp = np.array(imgnp * 255, dtype = np.uint8)
    
filename = os.path.splitext(os.path.basename(inputFileName))[0]
filePathAbs = output_dir + '/' + filename + '_out.jpeg'
print(filePathAbs)
printImage(imgnp, filePathAbs, 1, SAVE_WIDTH, SAVE_HEIGHT)

"""
# for a series of images=====================
# main loop
for count, file in enumerate(os.listdir(input_dir)): 
	out_string = str(count) + '/' + str(fileCount) + ': ' + file
	sys.stdout.write('%s\r' % out_string)
	sys.stdout.flush()
	
	inputFileName = file
	inputFilePath = input_dir + '/' + inputFileName
	input = loadImage(inputFilePath, 3, WIDTH, HEIGHT)	
	input *= 255
	input -= 127

	print("input_net shape:",input.shape)#input_net shape: (1, 3, 218, 298)
	output = testNet(net, input) 
	print("output_net shape:",output.shape)#output_net shape: (1, 1, 54, 74)
    
    	#rescale for output
	scaleW = float(SAVE_WIDTH) / float(OUT_WIDTH)
	scaleH = float(SAVE_HEIGHT) / float(OUT_HEIGHT)
                                        #output_net rescale: (1, 1, 218, 298)    
	output = scipy.ndimage.zoom(output, (1,1,scaleH,scaleW), order=3)
	output = ProcessToOutput(output)
	print("Scaled_output_net shape:",output.shape)


	imgnp = np.reshape(output, (SAVE_HEIGHT, SAVE_WIDTH, 1))#output shape: (218, 298, 1)
	imgnp = np.array(imgnp * 255, dtype = np.uint8)
    
	filename = os.path.splitext(os.path.basename(inputFileName))[0]
	filePathAbs = output_dir + '/' + filename + '.png'
	printImage(imgnp, filePathAbs, 1, SAVE_WIDTH, SAVE_HEIGHT)
"""
'''
for count, file in enumerate(os.listdir(output_dir)): 
	inputFilePath = output_dir + '/' + file
	imgnp = loadImage(inputFilePath, 1, 1920, 1080)
	imgnp = np.array(imgnp * 255, dtype = np.uint16)
	filePathAbs = output_dir + '/' + file 
	cv2.imwrite(filePathAbs, imgnp)
'''


