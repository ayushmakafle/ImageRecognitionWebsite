# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 17:23:50 2023

@author: amika
"""
from keras_preprocessing.image import img_to_array, ImageDataGenerator, load_img

X="dogs"
Y="cats"

sample_Y_image = "train/Y/cat.537.jpg"

datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rescale=1.0/255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

img=load_img(sample_Y_image)

x=img_to_array(img)
x=x.reshape((1,)+x.shape)

i=0

for batch in datagen.flow(x,
                          batch_size=1,
                          save_to_dir='preview',
                          save_prefix=Y,
                          save_format='jpg'):
    i += 1
    if i>20:
        break