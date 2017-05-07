# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 15:45:32 2016

@author: lihsintseng
"""
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.externals import joblib
from sklearn import svm
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

categories = ['Kitchen', 'Store', 'Bedroom', 'LivingRoom', 'Office',
       'Industrial', 'Suburb', 'InsideCity', 'TallBuilding', 'Street',
       'Highway', 'OpenCountry', 'Coast', 'Mountain', 'Forest'];
#categories = ['pos', 'neg'];
#feature_choosen = ['LBP','CoLBP','HOG','CoHOG','SIFT','CoSIFT','SURF','CoSURF']
#feature_choosen = ['CoLBP','CoHOG','CoSIFT','CoSURF','CoLBP+CoHOG','CoSIFT+CoSURF', 'CoLBP+CoHOG+CoSIFT+CoSURF']
#feature_choosen = ['CoLBP+CoHOG','CoSIFT+CoSURF', 'CoLBP+CoHOG+CoSIFT+CoSURF']
feature_choosen = ['CoLBP+CoHOG+CoSIFT+CoSURF']
#feature_choosen = ['SURF','SIFT+SURF']
#,'SIFT','SURF'

num_categories = len(categories);
dump_path = "/Users/lihsintseng/Desktop/Implementation/data/store/places/"
#dump_path = "/Users/lihsintseng/Desktop/Implementation/data/store/people/"
'''
for i in range(len(categories)):
	os.makedirs(dump_path+"confusion_matrices/"+categories[i])
'''
print("Loading features");
#import features
#train_lbp = joblib.load(dump_path+"Train_LBP/"+"train_lbp")
train_colbp = joblib.load(dump_path+"Train_CoLBP/"+"train_colbp")
#train_hog = joblib.load(dump_path+"Train_HOG/"+"train_hog")
train_cohog = joblib.load(dump_path+"Train_CoHOG/"+"train_cohog")
#train_brief = joblib.load(dump_path+"Train_BRIEF/"+"train_brief")
#train_cobrief = joblib.load(dump_path+"Train_CoBRIEF/"+"train_cobrief")
#train_sift = joblib.load(dump_path+"Train_SIFT/"+"train_sift")
train_cosift = joblib.load(dump_path+"Train_CoSIFT/"+"train_cosift")
#train_surf = joblib.load(dump_path+"Train_SURF/"+"train_surf")
train_cosurf = joblib.load(dump_path+"Train_CoSURF/"+"train_cosurf")
#test_lbp = joblib.load(dump_path+"Test_LBP/"+"test_lbp")
test_colbp = joblib.load(dump_path+"Test_CoLBP/"+"test_colbp")
#test_hog = joblib.load(dump_path+"Test_HOG/"+"test_hog")
test_cohog = joblib.load(dump_path+"Test_CoHOG/"+"test_cohog")
#test_brief = joblib.load(dump_path+"Test_BRIEF"+"test_brief")
#test_cobrief = joblib.load(dump_path+"Test_CoBRIEF"+"test_cobrief")
#test_sift = joblib.load(dump_path+"Test_SIFT/"+"test_sift")
test_cosift = joblib.load(dump_path+"Test_CoSIFT/"+"test_cosift")
#test_surf = joblib.load(dump_path+"Test_SURF/"+"test_surf")
test_cosurf = joblib.load(dump_path+"Test_CoSURF/"+"test_cosurf")
print("Loading Labels");
#import labels
train_labels = joblib.load(dump_path+"Train_Labels/"+"train_labels")
test_labels = joblib.load(dump_path+"Test_Labels/"+"test_labels")	
#choose features	

#train_feature_pile = [train_lbp,train_colbp,train_hog,train_cohog,
#					train_sift,train_cosift,train_surf,train_cosurf]
#test_feature_pile = [test_lbp,test_colbp,test_hog,test_cohog,
#					test_sift,test_cosift,test_surf,test_cosurf]

train_feature_pile = [np.hstack([train_colbp,train_cohog,train_cosift,train_cosurf])]
test_feature_pile = [np.hstack([test_colbp,test_cohog,test_cosift,test_cosurf])]

#train_feature_pile = [train_surf,np.hstack([train_sift,train_surf])]
#test_feature_pile = [test_surf,np.hstack([test_sift,test_surf])]

#train_feature_pile = [train_lbp,train_hog,
#					train_sift,train_surf]
#test_feature_pile = [test_lbp,test_hog,
#					test_sift,test_surf]
					
for i in range(len(feature_choosen)):
	print(feature_choosen[i]);
	train_features = train_feature_pile[i]
	#train_features = np.hstack([])
	test_features = test_feature_pile[i]
	#test_features = np.hstack([])
	#training
	
	LabelCompare = []
	LabelPredict = []
	
	#clf = AdaBoostClassifier()
	clf = svm.LinearSVC()
	
	Train_set = train_features[:]
	Train_Label = np.hstack(train_labels[:])
	Test_set = test_features[:]
	Test_Label = np.hstack(test_labels[:])
	
	#Training
	
	clf.fit(Train_set, Train_Label)
	PredictResult = clf.predict(Test_set)
	LabelCompare.append(Test_Label)
	LabelPredict.append(PredictResult)
	
	LabelCompare=np.array(LabelCompare)
	LabelCompare=np.hstack(LabelCompare)
	LabelPredict=np.array(LabelPredict)
	LabelPredict=np.hstack(LabelPredict)
	
	cm = confusion_matrix(LabelCompare,LabelPredict)

	#import matplotlib
	#matplotlib.image.imsave(dump_path+feature_choosen[i]+'.jpg', cm)
	
	#save cm image
	plt.imshow(cm, cmap='jet', interpolation='nearest') #Needs to be in row,col order
     #Greys_r
	#0->black, max->white
	plt.savefig(dump_path+feature_choosen[i]+'_SVM.jpg')
	#save cm
'''
data = open(dump_path+"confusion_matrices/"+feature_choosen[i]+"_SVM.txt","
uar = 0;
for x in range(cm.shape[0]):
	print str(float(cm[x,x])/np.sum(cm[x,:]))+"\n"
	uar += float(cm[x,x])/np.sum(cm[x,:])
print "UAR : "+str(uar/cm.shape[0])+"\n\n"
'''
