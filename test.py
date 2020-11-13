import keras
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import os
import time

dataset = "data/cifar10png/"
test_path = dataset + "test"
modelname = "type_your_model_name_here"

labels_name = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

test_imagepaths = sorted(list(paths.list_images(test_path)))
random.seed(42)
random.shuffle(test_imagepaths)

def data_preprocessing(imagepaths):
    labels = []
    data = []
    for imagepath in imagepaths:
        # load the image, pre-process it, and store it in the data list
        image = cv2.imread(imagepath)
        image = cv2.resize(image, (32, 32))
        image = img_to_array(image)
        data.append(image)
        label = imagepath.split(os.path.sep)[-2]

        for i in range(len(labels_name)):
            if label == labels_name[i]:
                label = i

        labels.append(label)

    image_normalize_mean = [0.485, 0.456, 0.406]
    image_normalize_std = [0.229, 0.224, 0.225]

    # scale the raw pixel intensities to the range [0, 1] and Standardization
    data = np.array(data, dtype="float") / 255.0
    for d in data:
        for channel in range(3):
            d[:, :, channel] -= image_normalize_mean[channel]
            d[:, :, channel] /= image_normalize_std[channel]

    labels = np.array(labels)

    return data, labels

#%%
test_data, test_labels = data_preprocessing(test_imagepaths)
(testX, testY) = (test_data, test_labels)
testY = to_categorical(testY, num_classes=len(labels_name))

start_time = time.time()
model = keras.models.load_model(str(modelname) + ".h5")
results = model.evaluate(testX, testY, batch_size=128)
end_time = time.time()
time_dif = end_time - start_time
print("Test loss, Test acc: ", results)
print("Time Required is: ", time_dif)