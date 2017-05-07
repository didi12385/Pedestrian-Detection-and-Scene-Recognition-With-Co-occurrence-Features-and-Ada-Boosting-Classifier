# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 15:03:40 2016

@author: lihsintseng
"""
import numpy as np
import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')

import cv2
from skimage.feature import local_binary_pattern
from skimage.feature import hog
from skimage import exposure
from skimage.feature import greycomatrix

def glcm_flatten(glcm):
	res = []
	for i in range(glcm.shape[0]):
		for j in range(glcm.shape[1]):
			res.append(glcm[i][j][0][0])
	return np.array(res)
def image_flatten(x):
	res = []
	for i in range(x.shape[0]):
		for j in range(x.shape[1]):
			res.append(x[i][j])
	return np.array(res)

def feature_flatten(x):
	res = []
	for i in range(x.shape[0]):
		res.append(x[i])
	return np.array(res)
#feature initialization
		
#CoHaar

#CoLBP
# compute the Local Binary Pattern representation
# of the image, and then use the LBP representation
# to build the histogram of patterns
def calculate_lbp(img):
	lbp = local_binary_pattern(img, 8,1, method="uniform");
	coLBP= greycomatrix(lbp, [5], [0], 81, symmetric=True, normed=True)
	return [image_flatten(lbp),glcm_flatten(coLBP)]
	
#CoHOG
#fd, hog_image = hog(img, orientations=8, pixels_per_cell=(4, 4),cells_per_block=(1, 1), visualise=True)
def calculate_hog(img):
	fd, hog_image = hog(img, orientations=8, pixels_per_cell=(8, 8),
	                    cells_per_block=(1, 1), visualise=True)
	# Rescale histogram for better display
	hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, hog_image.max())) * 8
	coHOG = greycomatrix(hog_image_rescaled, [5], [0], 64, symmetric=True, normed=True)
	return [image_flatten(hog_image_rescaled),glcm_flatten(coHOG)]

#http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_brief/py_brief.html#brief
#BRIEF
def calculate_brief(img):
	star = cv2.FeatureDetector_create("STAR")
	# Initiate BRIEF extractor
	brief = cv2.DescriptorExtractor_create("BRIEF")
	# find the keypoints with STAR
	kp = star.detect(img,None)
	# compute the descriptors with BRIEF
	kp, des = brief.compute(img, kp)
	des /= 8
	coBRIEF = greycomatrix(des, [5], [0], 64, symmetric=True, normed=True)
	return [image_flatten(des),glcm_flatten(coBRIEF)]
	
#http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_sift_intro/py_sift_intro.html#sift-intro
#SIFT
def calculate_sift(img):
	sift = cv2.SIFT()
	kp, des = sift.detectAndCompute(img,None)
	des /= 8
	coSIFT = greycomatrix(des, [5], [0], 64, symmetric=True, normed=True)
	return [image_flatten(des),glcm_flatten(coSIFT)]
	
#http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_surf_intro/py_surf_intro.html#surf
#SURF
# Create SURF object. You can specify params here or later.
# Here I set Hessian Threshold to 400
def calculate_surf(img):
	surf = cv2.SURF(400)
	# Find keypoints and descriptors directly
	kp, des = surf.detectAndCompute(img,None)
	des = des*4 + 4
	coSURF = greycomatrix(des, [5], [0], 64, symmetric=True, normed=True)
	return [image_flatten(des),glcm_flatten(coSURF)]