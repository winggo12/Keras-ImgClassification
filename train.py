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

dataset = "data/cifar10png/"
train_path = dataset + "train"
val_path = dataset + "val"

modelname = "type_your_model_name_here"
plot = "type_your_plot_name_here.png"


EPOCHS = 10
INIT_LR = 1e-3
LR_W_EXPODECAY = optimizers.schedules.ExponentialDecay(
    INIT_LR,
    decay_steps=1000,
    decay_rate=0.95,
    staircase=True)
BS = 32



#data = []
#labels = []
labels_name = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


train_imagepaths = sorted(list(paths.list_images(train_path)))
val_imagepaths = sorted(list(paths.list_images(val_path)))
random.seed(42)
random.shuffle(train_imagepaths)
random.shuffle(val_imagepaths)
#print(imagepaths)

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


train_data, train_labels = data_preprocessing(train_imagepaths)
val_data, val_labels = data_preprocessing(val_imagepaths)


# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
#(trainX, testX, trainY, testY) = train_test_split(data,labels, test_size=0.25, random_state=42)
(trainX, testX, trainY, testY) = (train_data, val_data, train_labels, val_labels)

# convert the labels from integers to vectors
trainY = to_categorical(trainY, num_classes=len(labels_name))
testY = to_categorical(testY, num_classes=len(labels_name))


# construct the image generator for data augmentation
# aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
# 	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
# 	horizontal_flip=True, fill_mode="nearest")

data_generator = ImageDataGenerator()

#Start Building the Model Right Now:

from keras import models
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D , AveragePooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers.core import Dropout


model = models.Sequential()

height = 32
width = 32
depth = 3
classes = len(labels_name)
inputShape = (height, width, depth)

model.add(Conv2D(filters=6, kernel_size=(5, 5), strides=(1, 1), padding='valid', activation='relu', input_shape=(32,32,3)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2) , padding='valid'))
model.add(Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1), padding='valid', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2) ,padding='valid'))
model.add(Flatten())
model.add(Dense(units=120, activation='relu'))
model.add(Dense(units=84, activation='relu'))
model.add(Dense(units=10, activation = 'softmax'))

opt = Adam(learning_rate=LR_W_EXPODECAY)

model.compile(loss="categorical_crossentropy", optimizer=opt,metrics=["accuracy"])


H = model.fit_generator(data_generator.flow(trainX, trainY, batch_size=BS),
	validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
	epochs=EPOCHS, verbose=1)


model.save(str(modelname) + ".h5")


# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_accuracy")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_accuracy")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(plot)


# Result of the accuracy using categorical_crossentropy (loss)


# load the image
# image = cv2.imread("test_images/santa_pepe.jpg")
# plt.imshow(image)
# plt.show()
# orig = image.copy()


# pre-process the image for classification
# image = cv2.resize(image, (28, 28))
# plt.imshow(image)
# image = image.astype("float") / 255.0
# image = img_to_array(image)
# image = np.expand_dims(image, axis=0)


# prob = model.predict(image)[0]
# print(model.predict(image)[0])


# maximum_prob = np.amax(model.predict(image)[0])
# predicted_class_number = np.where(prob == maximum_prob)[0][0]
# predicted_class = labels_name[predicted_class_number]
# print("The Prediction : " + predicted_class)



