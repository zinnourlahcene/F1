import numpy as np
import os
import scipy.ndimage
from skimage.feature import hog
from skimage import data, color, exposure
from sklearn.model_selection import  train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib
import matplotlib.pyplot as mpplot
import matplotlib.image as mpimg
from random import shuffle
import cv2
from PIL import Image
from sklearn.metrics import accuracy_score
import joblib


DIGITS_PATH = r'C:\Users\Dell\Desktop\F1\digits'

dataset = []
images = []
degits = []
val = -1
for root, _, folders in os.walk(DIGITS_PATH):
    current_directory_path = os.path.abspath(root)
    print(current_directory_path)
    for f in folders:
        current_image_path = os.path.join(current_directory_path, f)
        current_image = cv2.imread(current_image_path)
        current_image = current_image.reshape(1,-1)
        current_image = np.append(current_image, 0)
        images = current_image
        degits = val
        dataset.append([images, degits])

    val += 1

shuffle(dataset)

X = []
Y = []

for v in dataset:
    X.append(v[0])
    Y.append(v[1])

    
X_train, x_test, Y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)


knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, Y_train)
# get the model accuracy
model_score = knn.score(x_test, y_test)

# save trained model
joblib.dump(knn, 'knn_speed_model.pkl')
pred = knn.predict(x_test)

# evaluate accuracy
print(accuracy_score(y_test, pred))