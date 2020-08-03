import os
import chart_studio.plotly as py
import plotly.graph_objs as go
from PIL import Image, ImageDraw
from numpy import asarray
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.patches as patches
from os import listdir
from os.path import isfile, join
import itertools
import random
import pickle
import tensorflow as tf
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers
from keras.callbacks import LearningRateScheduler
from keras.utils import np_utils
from keras.utils import to_categorical
from numpy import load
from keras.callbacks import ModelCheckpoint

# Hyer-parameters
batch_size = 64
weight_decay = 1e-4 #coefficient keeping network simple
EPOCH_NUMBER = 30

#Data generator
class Snips_Generator(keras.utils.Sequence) :
  def __init__(self, image_filenames, labels, batch_size) :
    self.image_filenames = image_filenames
    self.labels = labels
    self.batch_size = batch_size
    
  def __len__(self) :
    return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(np.int)
  
  def __getitem__(self, idx) :
    batch_x = self.image_filenames[idx * self.batch_size : (idx+1) * self.batch_size]
    batch_y = self.labels[idx * self.batch_size : (idx+1) * self.batch_size]
    return np.array([load(str('data/'+file_name.replace('\\','/'))).astype('float32') for file_name in batch_x])/255.0 , np.array(batch_y)

#learning schedule
def lr_schedule(epoch):
    lrate = 0.001
    if epoch > 20:
        lrate = 0.0005
    if epoch > 25:
        lrate = 0.0003
    return lrate

#Creating instance of data generator
file_names_train = load('batches/file_names_train.npy')
file_names_test = load('batches/file_names_test.npy')

y_train = to_categorical(load('batches/y_train.npy'))
y_test = to_categorical(load('batches/y_test.npy'))

#Total number of categories = 2 = binary
number_classes = int(np.max(y_train)+1)

#Creating generators of batches
train_batch_generator = Snips_Generator(file_names_train, y_train, batch_size)
test_batch_generator = Snips_Generator(file_names_test, y_test, batch_size)

number_train_data = len(file_names_train)
print('Number of training data images: ',number_train_data)

number_test_data = len(file_names_test)
print('Number of test data images: ',number_test_data)

# Loading sample image to get input shape
sample_image = load(str(file_names_train[0].replace('\\','/'))).astype('float32')/255.0

#creating model
model=Sequential()
#First layer - Convolutional, Normalization, pooling, and dropout
model.add(Conv2D(16, (3,3), padding='same',  activation="relu", kernel_regularizer=regularizers.l2(weight_decay), input_shape=sample_image.shape))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))
#Second layer - Convolutional, Normalization, pooling, and dropout. 
model.add(Conv2D(32, (3,3), padding='same',  activation="relu", kernel_regularizer=regularizers.l2(weight_decay)))
model.add(BatchNormalization())
model.add(Conv2D(32, (3,3), padding='same',  activation="relu", kernel_regularizer=regularizers.l2(weight_decay)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))
#Third layer - Convolutional, Normalization, pooling, and dropout. 
model.add(Conv2D(64, (3,3), padding='same',  activation="relu", kernel_regularizer=regularizers.l2(weight_decay)))
model.add(BatchNormalization())
model.add(Conv2D(64, (3,3), padding='same',  activation="relu", kernel_regularizer=regularizers.l2(weight_decay)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))
#Fourth layer - Convolutional, Normalization, pooling, and dropout. 
model.add(Conv2D(128, (3,3), padding='same',  activation="relu", kernel_regularizer=regularizers.l2(weight_decay)))
model.add(BatchNormalization())
model.add(Conv2D(128, (3,3), padding='same',  activation="relu", kernel_regularizer=regularizers.l2(weight_decay)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))
#Fifth layer - Convolutional, Normalization, pooling, and dropout. 
model.add(Conv2D(256, (3,3), padding='same',  activation="relu", kernel_regularizer=regularizers.l2(weight_decay)))
model.add(BatchNormalization())
model.add(Conv2D(256, (3,3), padding='same',  activation="relu", kernel_regularizer=regularizers.l2(weight_decay)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))
#Sixth layer - Convolutional, Normalization, pooling, and dropout. 
model.add(Conv2D(512, (3,3), padding='same',  activation="relu", kernel_regularizer=regularizers.l2(weight_decay)))
model.add(BatchNormalization())
model.add(Dense(number_classes, activation='softmax'))

#training
checkpoint = ModelCheckpoint('model_checkpoint_CNN.h5', monitor='loss', verbose=1, save_best_only=True, mode='min')
model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.RMSprop(lr=0.001,decay=1e-6), metrics=['accuracy'])
history=model.fit_generator(generator=train_batch_generator,\
                    steps_per_epoch=int( number_train_data // batch_size),epochs=EPOCH_NUMBER,\
                    verbose=1,validation_data=test_batch_generator,validation_steps = int(number_test_data // batch_size),callbacks=[checkpoint,LearningRateScheduler(lr_schedule)])

#testing
scores = model.evaluate_generator(generator = test_batch_generator, steps = int(number_test_data // batch_size), verbose=1)
print('\nTest result: %.3f loss: %.3f' % (scores[1]*100,scores[0]))