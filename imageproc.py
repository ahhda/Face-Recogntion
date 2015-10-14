import cv2, os
from numpy import *
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA, RandomizedPCA
from sklearn.grid_search import GridSearchCV
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
from sklearn.svm import SVC

cascadeLocation = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadeLocation)

def prepare_dataset(directory):
	paths = [os.path.join(directory, filename) for filename in os.listdir(directory) if "sad" not in filename]
	images = []
	labels = []
	row = 140
	col = 140
	for image_path in paths:
		image_pil = Image.open(image_path).convert('L')
		image = np.array(image_pil, 'uint8')
		nbr = int(os.path.split(image_path)[1].split('.')[0].replace("subject",""))
		faces = faceCascade.detectMultiScale(image)
		for (x,y,w,h) in faces:
			images.append(image[y:y+col,x:x+row])
			labels.append(nbr)
			cv2.imshow("Reading Faces ",image[y:y+col,x:x+row])
			cv2.waitKey(50)
	return images,labels, row, col

directory = 'yalefaces'
images, labels, row, col = prepare_dataset(directory)
n_components = 10
cv2.destroyAllWindows()
pca = RandomizedPCA(n_components=n_components, whiten=True)

param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
clf = GridSearchCV(SVC(kernel='rbf', class_weight='auto'),param_grid)

testing_data = []
for i in range(len(images)):
	testing_data.append(images[i].flatten())
pca = pca.fit(testing_data)

transformed = pca.transform(testing_data)
clf.fit(transformed,labels)

image_paths = [os.path.join(directory, filename) for filename in os.listdir(directory) if filename.endswith('sad')]
for image_path in image_paths:
	pred_image_pil = Image.open(image_path).convert('L')
	pred_image = np.array(pred_image_pil, 'uint8')
	faces = faceCascade.detectMultiScale(pred_image)
	for (x,y,w,h) in faces:
		X_test = pca.transform(np.array(pred_image[y:y+col,x:x+row]).flatten())
		mynbr = clf.predict(X_test)
		nbr_act = int(os.path.split(image_path)[1].split('.')[0].replace("subject",""))
		print "Predicted By Classifier : ",mynbr[0], " Actual : ", nbr_act
