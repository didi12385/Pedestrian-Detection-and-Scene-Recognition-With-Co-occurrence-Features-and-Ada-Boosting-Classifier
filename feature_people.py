# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 21:55:02 2016

@author: lihsintseng
"""

#Object detection with co-occurrence features(CoHaar, CoLBP,CoHOG)
#Using AdaBoostClassifier to train and predict

import numpy as np
import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')
sys.path.insert(0, '/Users/lihsintseng/Desktop/Implementation/')
import cv2
import scipy
import glob
from sklearn.externals import joblib
import feature_function
#
categories = ['pos', 'neg'];
num_categories = len(categories);
dump_path = "/Users/lihsintseng/Desktop/Implementation/data/store/people/"
train_image_paths = [];
train_labels = [];
test_image_paths = [];
test_labels = [];

new = [];
#try one
#new.append("/Users/lihsintseng/Desktop/Implementation/data/places/train/Store/image_0005.jpg");

#import all training&testing images
for i in range(num_categories):
    
    if i == 0: path = "/Users/lihsintseng/Documents/FirstYear/AI/INRIAPerson/train_64x128_H96/pos/";
    else: path = "/Users/lihsintseng/Documents/FirstYear/AI/INRIAPerson/Train/neg/";
    
    names = glob.glob(path+'*.png');
    if glob.glob(path+'*.jpg')!= []: names+=glob.glob(path+'*.jpg')
    new = [];
    labels = [];
    for j in range(len(names)):
        new.append(names[j]);
        labels.append(i);								
        #labels.append(categories[i]);
								
    train_image_paths.append(new);
    train_labels.append(labels);
    
    if i == 0: path = "/Users/lihsintseng/Documents/FirstYear/AI/INRIAPerson/test_64x128_H96/pos/";
    else: path = "/Users/lihsintseng/Documents/FirstYear/AI/INRIAPerson/Test/neg/";
    
    names = glob.glob(path+'*.png');
    if glob.glob(path+'*.jpg')!= []: names+=glob.glob(path+'*.jpg')
	
    new = [];
    labels = [];
    for j in range(len(names)):
        new.append(names[j])
        labels.append(i);
        #labels.append(categories[i]);
    test_image_paths.append(new);
    test_labels.append(labels);
				
#os.makedirs(dump_path+"Train_Labels")
joblib.dump(train_labels, dump_path+"Train_Labels/"+"train_labels")
#os.makedirs(dump_path+"Test_Labels")
joblib.dump(test_labels, dump_path+"Test_Labels/"+"test_labels")

#image preprocess
train_lbp = []
train_colbp = []
train_hog = []
train_cohog = []
train_sift = []
train_cosift = []
train_surf = []
train_cosurf = []

for i in range(num_categories):
#i=1;
	print("currently extracting from training files : "+categories[i]);
	for j in range(len(train_image_paths[i])):
		print(j);
		img = cv2.imread(train_image_paths[i][j],0)
		img = scipy.misc.imresize(img,(64,128))
		imgdx = cv2.Sobel( img, cv2.CV_16SC1, 1, 0, 3 );
		imgdx = cv2.convertScaleAbs(imgdx);
		imgdy = cv2.Sobel( img, cv2.CV_16SC1, 0, 1, 3 );
		imgdy = cv2.convertScaleAbs(imgdy)
		'''
		res = feature_function.calculate_lbp(img)
		resx = feature_function.calculate_lbp(imgdx)
		resy = feature_function.calculate_lbp(imgdy)
		lbp = np.hstack([res[0],resx[0],resy[0]])
		colbp = np.hstack([res[1],resx[1],resy[1]])
		train_lbp.append(lbp)
		train_colbp.append(colbp)
		'''
		res = feature_function.calculate_hog(img)
		resx = feature_function.calculate_hog(imgdx)
		resy = feature_function.calculate_hog(imgdy)
		hog = np.hstack([res[0],resx[0],resy[0]])
		cohog = np.hstack([res[1],resx[1],resy[1]])
		train_hog.append(hog)
		train_cohog.append(cohog)
		'''
		res = feature_function.calculate_sift(img)
		resx = feature_function.calculate_sift(imgdx)
		resy = feature_function.calculate_sift(imgdy)
		sift = np.hstack([res[0],resx[0],resy[0]])
		cosift = np.hstack([res[1],resx[1],resy[1]])
		train_sift.append(sift)
		train_cosift.append(cosift)
		
		res = feature_function.calculate_surf(img)
		resx = feature_function.calculate_surf(imgdx)
		resy = feature_function.calculate_surf(imgdy)
		surf = np.hstack([res[0],resx[0],resy[0]])
		cosurf = np.hstack([res[1],resx[1],resy[1]])
		train_surf.append(surf)
		train_cosurf.append(cosurf)
		'''
'''
joblib.dump(train_lbp,dump_path+"Train_LBP/"+"train_lbp")
joblib.dump(train_colbp,dump_path+"Train_CoLBP/"+"train_colbp")
'''
joblib.dump(train_hog,dump_path+"Train_HOG/"+"train_hog")
joblib.dump(train_cohog,dump_path+"Train_CoHOG/"+"train_cohog")
'''
joblib.dump(train_sift,dump_path+"Train_SIFT/"+"train_sift")
joblib.dump(train_cosift,dump_path+"Train_CoSIFT/"+"train_cosift")
joblib.dump(train_surf,dump_path+"Train_SURF/"+"train_surf")
joblib.dump(train_cosurf,dump_path+"Train_CoSURF/"+"train_cosurf")
'''


test_lbp = []
test_colbp = []
test_hog = []
test_cohog = []
test_sift = []
test_cosift = []
test_surf = []
test_cosurf = []
for i in range(num_categories):
	print("currently extracting from testing files : "+categories[i]);
	#i=1;
	for j in range(len(test_image_paths[i])):
		print(j);
		img = cv2.imread(test_image_paths[i][j],0);
		img = scipy.misc.imresize(img,(64,128));
		imgdx = cv2.Sobel( img, cv2.CV_16SC1, 1, 0, 3 );
		imgdx = cv2.convertScaleAbs(imgdx);
		imgdy = cv2.Sobel( img, cv2.CV_16SC1, 0, 1, 3 );
		imgdy = cv2.convertScaleAbs(imgdy)
		
		'''
		res = feature_function.calculate_lbp(img)
		resx = feature_function.calculate_lbp(imgdx)
		resy = feature_function.calculate_lbp(imgdy)
		lbp = np.hstack([res[0],resx[0],resy[0]])
		colbp = np.hstack([res[1],resx[1],resy[1]])
		test_lbp.append(lbp)
		test_colbp.append(colbp)
		'''
		res = feature_function.calculate_hog(img)
		resx = feature_function.calculate_hog(imgdx)
		resy = feature_function.calculate_hog(imgdy)
		hog = np.hstack([res[0],resx[0],resy[0]])
		cohog = np.hstack([res[1],resx[1],resy[1]])
		test_hog.append(hog)
		test_cohog.append(cohog)
		'''
		res = feature_function.calculate_sift(img)
		resx = feature_function.calculate_sift(imgdx)
		resy = feature_function.calculate_sift(imgdy)
		sift = np.hstack([res[0],resx[0],resy[0]])
		cosift = np.hstack([res[1],resx[1],resy[1]])
		test_sift.append(sift)
		test_cosift.append(cosift)
		
		res = feature_function.calculate_surf(img)
		resx = feature_function.calculate_surf(imgdx)
		resy = feature_function.calculate_surf(imgdy)
		surf = np.hstack([res[0],resx[0],resy[0]])
		cosurf = np.hstack([res[1],resx[1],resy[1]])
		test_surf.append(surf)
		test_cosurf.append(cosurf)
		'''
'''
joblib.dump(test_lbp,dump_path+"Test_LBP/"+"test_lbp")
joblib.dump(test_colbp,dump_path+"Test_CoLBP/"+"test_colbp")
'''
joblib.dump(test_hog,dump_path+"Test_HOG/"+"test_hog")
joblib.dump(test_cohog,dump_path+"Test_CoHOG/"+"test_cohog")
'''
joblib.dump(test_sift,dump_path+"Test_SIFT/"+"test_sift")
joblib.dump(test_cosift,dump_path+"Test_CoSIFT/"+"test_cosift")
joblib.dump(test_surf,dump_path+"Test_SURF/"+"test_surf")
joblib.dump(test_cosurf,dump_path+"Test_CoSURF/"+"test_cosurf")
'''
