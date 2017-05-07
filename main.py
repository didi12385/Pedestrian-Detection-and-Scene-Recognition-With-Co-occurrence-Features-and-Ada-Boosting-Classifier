# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 16:40:44 2016

@author: lihsintseng
"""
#Object detection with co-occurrence features(CoHaar, CoLBP,CoHOG)
#Using AdaBoostClassifier to train and predict

import numpy as np
import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')
import cv2
import os
from skimage.feature import local_binary_pattern
from skimage.feature import hog
from skimage import data, color, exposure
from skimage.feature import greycomatrix, greycoprops
from sklearn.ensemble import AdaBoostClassifier



categories = ['Kitchen', 'Store', 'Bedroom', 'LivingRoom', 'Office',
       'Industrial', 'Suburb', 'InsideCity', 'TallBuilding', 'Street',
       'Highway', 'OpenCountry', 'Coast', 'Mountain', 'Forest'];
num_categories = len(categories);
num_train_per_cat = 100; 
data_path = "/Users/lihsintseng/Desktop/Implementation/data/";
train_image_paths = [];
train_labels = [];
test_image_paths = [];
test_labels = [];

new = [];
#try one
new.append("/Users/lihsintseng/Desktop/Implementation/data/train/Store/image_0005.jpg");
'''
#import all training&testing images
for i in range(num_categories):
    path = data_path + 'train/' + categories[i] + '/';
    names = os.listdir(path);
    new = [];
    labels = [];
    for j in range(len(names)):
        new.append(path+names[j]);
        labels.append(categories[i]);
    train_image_paths.append(new);
    train_labels.append(labels);
    
    path = data_path + 'test/' + categories[i] + '/';
    names = os.listdir(path);
    new = [];
    labels = [];
    for j in range(len(names)):
        new.append(path+names[j])
        labels.append(categories[i]);
    test_image_paths.append(new);
    test_labels.append(labels);
    '''


#image preprocess


img = cv2.imread(new[0],0)
imgdx = cv2.Sobel( img, cv2.CV_16SC1, 1, 0, 3 );
imgdy = cv2.Sobel( img, cv2.CV_16SC1, 0, 1, 3 );
#feature initialization

#CoHaar

#CoLBP

# compute the Local Binary Pattern representation
# of the image, and then use the LBP representation
# to build the histogram of patterns
lbp = local_binary_pattern(img, 8,1, method="uniform");
'''
mod = np.zeros(9);
for i in range(lbp.shape[0]):
	for j in range(lbp.shape[1]):
		temp = 0;
		while(lbp[i][j]/2>=1):
		    if(lbp[i][j]%2==1):
		        temp+=1;
		        lbp[i][j]-=1;
		    lbp[i][j]/=2;
		temp+=1;
		mod[temp]+=1;
'''	
(hist, _) = np.histogram(lbp.ravel(),bins=np.arange(0, 10),range=(0, 9));

# normalize the histogram
#hist = hist.astype("float")
#hist /= (hist.sum() + 1e-7)
glcm = greycomatrix(lbp, [5], [0], 81, symmetric=True, normed=True)
#CoHOG

#fd, hog_image = hog(img, orientations=8, pixels_per_cell=(4, 4),cells_per_block=(1, 1), visualise=True)
fd, hog_image = hog(img, orientations=8, pixels_per_cell=(8, 8),
                    cells_per_block=(1, 1), visualise=True)

# Rescale histogram for better display
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02)) * 8
glcm = greycomatrix(hog_image_rescaled, [5], [0], 64, symmetric=True, normed=True)

#http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_brief/py_brief.html#brief
#BRIEF
star = cv2.FeatureDetector_create("STAR")
# Initiate BRIEF extractor
brief = cv2.DescriptorExtractor_create("BRIEF")
# find the keypoints with STAR
kp = star.detect(img,None)
# compute the descriptors with BRIEF
kp, des = brief.compute(img, kp)
des /= 8
glcm = greycomatrix(des, [5], [0], 64, symmetric=True, normed=True)
#http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_sift_intro/py_sift_intro.html#sift-intro
#SIFT
sift = cv2.SIFT()
kp, des = sift.detectAndCompute(img,None)
des /= 8
glcm = greycomatrix(des, [5], [0], 64, symmetric=True, normed=True)

#http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_surf_intro/py_surf_intro.html#surf
#SURF
# Create SURF object. You can specify params here or later.
# Here I set Hessian Threshold to 400
surf = cv2.SURF(400)
# Find keypoints and descriptors directly
kp, des = surf.detectAndCompute(img,None)
des = des*4 + 4
glcm = greycomatrix(des, [5], [0], 64, symmetric=True, normed=True)


#training
Train_set = []
Train_Label = []
Test_set = []
Test_Label = []

for i in range(num_categories):
	if i ==0:
		Train_set = np.hstack(am[1:])
		Train_Label = np.hstack(emo[1:])
		Test_set = np.hstack(am[1:])
		Test_Label = np.array(emo[i])
	elif i == 9:
		Train_set = np.hstack(am[:i])
		Train_Label = np.hstack(emo[:i])
		Test_set = np.array(am[i])
		Test_Label = np.array(emo[i])
	else:
		Train_set = np.concatenate(( np.hstack(am[:i ]), np.hstack(am[ i+1:]) ))
		Train_Label = np.concatenate(( np.hstack(emo[:i ]), np.hstack(emo[ i+1:]) ))
		Test_set = np.array(am[i])
		Test_Label = np.array(emo[i])


#Training
'''
clf = AdaBoostClassifier()
clf.fit()
clf.predict()
'''