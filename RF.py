import glob
import cv2
import numpy as np

from PIL import Image
import PIL

from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from skimage.feature import hog
import mahotas.features.surf as surf


def load_data(extension, ext):

    path = glob.glob("C:/Users/Arhum/Desktop/Assignments/Machine Learning/Assignment-1/archive/data/"+extension+"/*."+ext)
    
    data_sift = []
    data_hog = []
    data_surf = []
    
    for file in path:
        img = cv2.imread(file)
        
        img = cv2.resize(img,(128,128))
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        
        #SIFT
        sift = cv2.SIFT_create(nfeatures=500) 
        
        keypoints , descriptors = sift.detectAndCompute(img, None)
        
        # Flatten descriptors
        descriptors_flat = np.zeros((500, 128))
        descriptors_flat[:descriptors.shape[0], :] = descriptors
        
        data_sift.append(descriptors_flat.flatten())
        
        
        #HOG
        fd, hog_image = hog(img, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True, multichannel=False)
        
        data_hog.append(fd)
        
        
        #SURF
        fd = surf.surf(img ,descriptor_only=True)
        
        fd = fd[:min(len(fd), 500), :]
        
        fd = np.pad(fd, ((0, 500 - fd.shape[0]), (0, 0)), 'constant')
        
        data_surf.append(fd)

    return np.array(data_sift), np.array(data_hog), np.array(data_surf)


labels = [ 'cats', 'jpg', 'dogs', 'jpg', 'horses','jpg']

# Load the data for each category
cats = load_data(labels[0], labels[1])
dogs = load_data(labels[2], labels[3])
horses = load_data(labels[4], labels[5])


# Concatenate the data for all categories
data_set_sift = np.concatenate((cats[0],dogs[0],horses[0]), axis=0)
data_set_hog = np.concatenate((cats[1],dogs[1],horses[1]), axis=0)
data_set_surf = np.concatenate((cats[2],dogs[2],horses[2]), axis=0)

data_set_surf = np.array([np.concatenate(data_set_surf[i]) for i in range(len(data_set_surf))])

rows = data_set_sift.shape[0]
    
labels = np.zeros((rows))

labels[0:202] = 0
labels[202:404] = 1
labels[404:606] = 2


#------------- SIFT
#Splitting
x_train, x_test, y_train, y_test = train_test_split(data_set_sift, labels, test_size=0.3)

#Model 
model = RandomForestClassifier()
model.fit(x_train, y_train)

y_predict = model.predict(x_test)

# Evaluation
acc = accuracy_score(y_test, y_predict)
score = model.score(x_test, y_test)
print('SIFT Accuracy:',acc)
print('SIFT Score:', score)


#------------- HOG
#Splitting
x_train, x_test, y_train, y_test = train_test_split(data_set_hog, labels, test_size=0.3)

#Model 
model = RandomForestClassifier()
model.fit(x_train, y_train)

y_predict = model.predict(x_test)

# Evaluation
acc = accuracy_score(y_test, y_predict)
score = model.score(x_test, y_test)
print('\nHOG Accuracy:',acc)
print('HOG Score:', score)



#------------- SURF
#Splitting
x_train, x_test, y_train, y_test = train_test_split(data_set_surf, labels, test_size=0.3)

#Model 
model = RandomForestClassifier()
model.fit(x_train, y_train)

y_predict = model.predict(x_test)

# Evaluation
acc = accuracy_score(y_test, y_predict)
score = model.score(x_test, y_test)
print('\nSURF Accuracy:',acc)
print('SURF Score:', score)