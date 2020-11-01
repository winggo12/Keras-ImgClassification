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
classes = 10
inputShape = (height, width, depth)

model.add(Conv2D(filters=6, kernel_size=(5, 5), strides=(1, 1), padding='valid', activation='relu', input_shape=(32,32,3)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2) , padding='valid'))

model.add(Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1), padding='valid', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2) ,padding='valid'))

model.add(Flatten())

model.add(Dense(units=120, activation='relu'))

model.add(Dense(units=84, activation='relu'))

model.add(Dense(units=10, activation = 'softmax'))

# opt = Adam(lr=LR_W_EXPODECAY)

print(model.summary())

model.compile(loss="categorical_crossentropy", optimizer=opt,metrics=["accuracy"])