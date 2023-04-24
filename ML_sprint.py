import numpy as np
from keras.layers import Input, Dense, Activation,BatchNormalization, Flatten, Conv2D, MaxPooling2D
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

path = "C:\\Users\\yuvan\\OneDrive\\Desktop\\Yuvi\\VITB_hackathon\\seg_train\\seg_train"
train_datagen = ImageDataGenerator(rescale=1. / 255)
train = train_datagen.flow_from_directory(path, target_size=(227,227), class_mode='categorical')

type(train)
type(train_datagen)

'''
print("Batch Size for Input Image : ", train[0][0].shape)
print("Batch Size for Output Image : ", train[0][1].shape)
print("Image Size of first image : ", train[0][0][0].shape)
print("Output of first image : ", train[0][1][0].shape)
'''

fig, axs = plt.subplots(2, 3, figsize=(10, 10))

axs[0][0].imshow(train[0][0][12])
axs[0][0].set_title(train[0][1][12])

axs[0][1].imshow(train[0][0][10])
axs[0][1].set_title(train[0][1][10])

axs[0][2].imshow(train[0][0][5])
axs[0][2].set_title(train[0][1][5])

axs[1][0].imshow(train[0][0][20])
axs[1][0].set_title(train[0][1][20])

axs[1][1].imshow(train[0][0][25])
axs[1][1].set_title(train[0][1][25])

axs[1][2].imshow(train[0][0][3])
axs[1][2].set_title(train[0][1][3])


def Alexnet(input_shape):
    X_input = Input(input_shape)

    X = Conv2D(96, (11, 11), strides=4, name="conv0")(X_input)
    X = BatchNormalization(axis=3, name="bn0")(X)
    X = Activation('relu')(X)

    X = MaxPooling2D((3, 3), strides=2, name='max0')(X)

    X = Conv2D(256, (5, 5), padding='same', name='conv1')(X)
    X = BatchNormalization(axis=3, name='bn1')(X)
    X = Activation('relu')(X)

    X = MaxPooling2D((3, 3), strides=2, name='max1')(X)

    X = Conv2D(384, (3, 3), padding='same', name='conv2')(X)
    X = BatchNormalization(axis=3, name='bn2')(X)
    X = Activation('relu')(X)

    X = Conv2D(384, (3, 3), padding='same', name='conv3')(X)
    X = BatchNormalization(axis=3, name='bn3')(X)
    X = Activation('relu')(X)

    X = Conv2D(256, (3, 3), padding='same', name='conv4')(X)
    X = BatchNormalization(axis=3, name='bn4')(X)
    X = Activation('relu')(X)

    X = MaxPooling2D((3, 3), strides=2, name='max2')(X)

    X = Flatten()(X)

    X = Dense(4096, activation='relu', name="fc0")(X)

    X = Dense(4096, activation='relu', name='fc1')(X)

    X = Dense(6, activation='softmax', name='fc2')(X)

    model = Model(inputs=X_input, outputs=X, name='AlexNet')

    return model

predictions = alex.predict_generator(predict)
len(predictions)

imshow(predict[5300][0][0])
plt.imsave("predicted1.png", predict[5300][0][0])

print(predictions[5300])
imshow(predict[1300][0][0])

predictions[1300]
imshow(predict[4400][0][0])

predictions[4400]
imshow(predict[700][0][0])
plt.imsave("predicted2.png", predict[700][0][0])

print(predictions[700])
imshow(predict[6500][0][0])

predictions[6500]

import os
def get_category(predicted_output):
    path  = "C:\\Users\\yuvan\\OneDrive\\Desktop\\Yuvi\\VITB_hackathon\\seg_train\\seg_train"
    return os.listdir(path)[np.argmax(predicted_output)]
print(get_category(predictions[700]))

fig, axs = plt.subplots(2, 3, figsize=(10, 10))

axs[0][0].imshow(predict[1002][0][0])
axs[0][0].set_title(get_category(predictions[1002]))

axs[0][1].imshow(predict[22][0][0])
axs[0][1].set_title(get_category(predictions[22]))

axs[0][2].imshow(predict[1300][0][0])
axs[0][2].set_title(get_category(predictions[1300]))

axs[1][0].imshow(predict[3300][0][0])
axs[1][0].set_title(get_category(predictions[3300]))

axs[1][1].imshow(predict[7002][0][0])
axs[1][1].set_title(get_category(predictions[7002]))

axs[1][2].imshow(predict[512][0][0])
axs[1][2].set_title(get_category(predictions[512]))