#!/usr/bin/env python
# coding: utf-8

# In[11]:


import numpy as np
import tensorflow as tf


import sys



X_train = np.zeros((1,512,512,1), dtype=int)


from keras.layers import Input, Activation, Conv2D, MaxPool2D, UpSampling2D, Dropout, concatenate, BatchNormalization, Cropping2D, ZeroPadding2D, SpatialDropout2D
from keras.layers import Conv2DTranspose, Dropout, GaussianNoise
from keras.models import Model
from keras import backend as K


import keras.backend as K
from keras.optimizers import Adam
from keras.losses import binary_crossentropy ,mean_squared_error
def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1,2])
    union = K.sum(y_true, axis=[1,2]) + K.sum(y_pred, axis=[1,2])
    return K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)
def dice_p_bce(in_gt, in_pred):
    return 0.0*binary_crossentropy(in_gt, in_pred) - dice_coef(in_gt, in_pred)
def true_positive_rate(y_true, y_pred):
    return K.sum(K.flatten(y_true)*K.flatten(K.round(y_pred)))/K.sum(y_true)
#seg_model.load_weights('../input/weights/seg_model_weights.best.hdf5')
#seg_model.compile(optimizer=Adam(1e-4, decay=1e-6), loss=dice_p_bce, metrics=[dice_coef, 'binary_accuracy', true_positive_rate])
def up_scale(in_layer):
    filt_count = in_layer._keras_shape[-1]
    return Conv2DTranspose(filt_count, kernel_size = (2,2), strides = (2,2), padding = 'same')(in_layer)
def up_scale_fancy(in_layer):
    return UpSampling2D(size=(2,2))(in_layer)

input_layer = Input(shape=X_train.shape[1:])
sp_layer = GaussianNoise(0.1)(input_layer)
bn_layer = BatchNormalization()(sp_layer)
c1 = Conv2D(filters=8, kernel_size=(5,5), activation='relu', padding='same')(bn_layer)
l = MaxPool2D(strides=(2,2))(c1)
c2 = Conv2D(filters=16, kernel_size=(3,3), activation='relu', padding='same')(l)
l = MaxPool2D(strides=(2,2))(c2)
c3 = Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same')(l)

l = MaxPool2D(strides=(2,2))(c3)
c4 = Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same')(l)

l = SpatialDropout2D(0.25)(c4)
dil_layers = [l]
for i in [2, 4, 6, 8, 12, 18, 24]:
    dil_layers += [Conv2D(16,
                          kernel_size = (3, 3), 
                          dilation_rate = (i, i), 
                          padding = 'same',
                         activation = 'relu')(l)]
l = concatenate(dil_layers)


l = SpatialDropout2D(0.2)(concatenate([up_scale(l), c3], axis=-1))
l = Conv2D(filters=128, kernel_size=(2,2), activation='linear', padding='same')(l)
l = BatchNormalization()(l)
l = Activation('relu')(l)
u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (l)
l = Conv2D(filters=96, kernel_size=(2,2), activation='relu', padding='same')(u9)
l = Conv2D(filters=32, kernel_size=(2,2), activation='relu', padding='same')(l)
u10 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (l)
l = Conv2D(filters=16, kernel_size=(2,2), activation='linear', padding='same')(u10)

l = Cropping2D((4,4))(l)
l = BatchNormalization()(l)
l = Activation('relu')(l)

l = Conv2D(filters=1, kernel_size=(1,1), activation='sigmoid')(l)
output_layer = ZeroPadding2D((4,4))(l)

seg_model = Model(input_layer, output_layer)
seg_model.load_weights('C:/Users/Slahu/Downloads/fyp_mid/bestweight1.hdf5')
# sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
seg_model.compile(optimizer='sgd', loss=dice_p_bce, metrics=[tf.keras.metrics.MeanIoU(num_classes=2),dice_coef, 'binary_accuracy', true_positive_rate])
seg_model.summary()
#Adam(1e-4, decay=1e-6)






from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
import matplotlib.pyplot as plt
from PIL import Image


img = imread('C:/Users/Slahu/Downloads/U-Net/data/test/1/images/1.png')
print(img.shape)
img = resize(img, (512, 512, 1), mode='constant', preserve_range=True)
print(img.shape)




x_train = np.zeros((1,512,512,1), dtype=int)
y_train = np.zeros((1,512,512,1), dtype=int)
x_train[0] = img



fig, m_axs = plt.subplots(2,3, figsize = (20, 20))
[c_ax.axis('off') for c_ax in m_axs.flatten()]

ix = x_train
p_image = seg_model.predict(ix)

im = np.zeros((512,512), dtype=int)

for i in range(512):
    for j in range(512):
        im[i][j] = p_image[0,i,j,0]
  
I = Image.fromarray(im).save("r.png")


# In[ ]:




