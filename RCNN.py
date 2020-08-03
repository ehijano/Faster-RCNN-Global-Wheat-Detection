import datetime
import os
from os import listdir
from os.path import isfile, join
import chart_studio.plotly as py
import plotly.graph_objs as go
import numpy as np
import numpy.random as npr
from numpy import asarray
from numpy import load
import random
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import matplotlib.patches as patches
#ML imports
import tensorflow as tf
import keras
import keras.backend as K
from tensorflow.keras.layers import Layer
from keras.losses import MSE
from keras.models import Sequential
from keras.utils import np_utils, to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers
from keras.callbacks import LearningRateScheduler
import json
from keras.models import load_model
from keras.models import model_from_json
from keras.models import Model, Input
from keras.callbacks import LearningRateScheduler
from utils import huberGTZ, xywh_xyXY, extract_crop, generate_anchors, draw_anchors,bbox_overlaps,unmap,bbox_transform,loss_cls, smoothL1, bbox_transform_inv,filter_boxes,clip_boxes,py_cpu_nms
from random import seed
from random import randint
from keras.callbacks import ModelCheckpoint




def generate_proposals( data ):
  # Extract feature map
  feature_map = CNN_model_cut.predict(data.reshape(-1,data.shape[0],data.shape[1],data.shape[2]))
  padded_fcmap = np.pad(feature_map,((0,0),(1,1),(1,1),(0,0)),mode='constant') 

  # Extract RPN results
  RPN_results = RPN_model.predict(padded_fcmap)
  anchor_probs = RPN_results[0].reshape((-1,1))
  anchor_targets = RPN_results[1].reshape((-1,4))

  # Original anchors
  feature_size = feature_map.shape[1]
  number_feature_points = feature_size * feature_size
  feature_stride = int( image_size / feature_size )
  base_anchors = generate_anchors(feature_stride, feature_stride,ratios = ANCHOR_RATIOS, scales = ANCHOR_SCALES)
  shift = np.arange(0, feature_size) * feature_stride
  shift_x, shift_y = np.meshgrid(shift, shift)
  shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel())).transpose()
  original_anchors = (base_anchors.reshape((1, anchor_number, 4)) +shifts.reshape((1, number_feature_points, 4)).transpose((1, 0, 2)))
  original_anchors = original_anchors.reshape((-1, 4))

  # Proposals by the RPN
  proposals = bbox_transform_inv(original_anchors, anchor_targets)
  proposals = clip_boxes(proposals, (data.shape[0],data.shape[1]))  # clip to image. 
  high_to_low_scores = anchor_probs.ravel().argsort()[::-1] # highest scores
  high_to_low_scores = high_to_low_scores[0:N]
  proposals = proposals[high_to_low_scores,:]
  anchor_probs = anchor_probs[high_to_low_scores]

  del original_anchors
  del RPN_results
  del feature_map
  del padded_fcmap

  return proposals, anchor_probs



def produce_batch( image_file , true_boxes):
  image = Image.open(image_file).resize((image_size ,image_size ), Image.NEAREST)
  data =  asarray(image)/255.0
  del image

  proposals, anchor_probs = generate_proposals( data )
  del data

  # Non maximal suppression
  keep = py_cpu_nms(np.hstack((proposals , anchor_probs)), NSM_THRESHOLD)
  if post_nms_N > 0:
      keep = keep[:post_nms_N]
  proposals = proposals[keep, :]
  anchor_probs = anchor_probs[keep]


  # RCNN proposals
  #proposals = np.vstack( (proposals, true_boxes) )    
  overlaps = bbox_overlaps(proposals , enlarged_bboxes )
  which_box = overlaps.argmax(axis=1)
  proposal_max_overlaps = overlaps.max(axis=1)
  
  # sub sample foreground and background
  fg_inds = np.where(proposal_max_overlaps >= FG_THRESHOLD_RCNN )[0]
  fg_rois_in_image = min( int(BATCH_SIZE/(1+BG_FG_FRAC_RCNN))  , fg_inds.size )  
  if fg_inds.size > 0:
      fg_inds = npr.choice(fg_inds, size=fg_rois_in_image, replace=False)

  bg_inds = np.where((proposal_max_overlaps < BG_THRESH_HI) & (proposal_max_overlaps >= BG_THRESH_LO))[0]
  bg_rois_in_image = min(fg_rois_in_image, bg_inds.size)
  if bg_inds.size > 0:
      bg_inds = npr.choice(bg_inds, size=bg_rois_in_image, replace=False)

  keep_inds = np.append(fg_inds, bg_inds)
  np.random.shuffle(keep_inds)

  # Select sampled values from various arrays:
  rois = proposals[keep_inds] # The chosen rois
  # Scores of chosen rois (fg=1, bg=0)
  new_scores = np.zeros(len(proposals)) 
  new_scores[fg_inds] = 1 
  roi_scores = new_scores[keep_inds].reshape(-1,1) 
  # targets
  targets = np.zeros((len(proposals),4)).reshape(-1,4)
  targets[fg_inds] = bbox_transform( proposals[fg_inds],    true_boxes[which_box[fg_inds]]   )
  targets = targets[keep_inds]

  return rois, targets, roi_scores


def input_generator(filesDIR):
    batch_rois=[]
    batch_inds=[]
    batch_fmaps = []
    image_counter = 0

    batch_scores=[]
    batch_targets=[]
    scan_counter = 0
    while 1:
        print('scans: ', scan_counter)
        for f in listdir(filesDIR):
            scan_counter+=1
            data =  asarray(Image.open(filesDIR+f))/255.0

            feature_map_for_RoIPool = CNN_model_RoI.predict(data.reshape(-1,data.shape[0],data.shape[1],data.shape[2]))
            feature_stride_for_ROIPool = int( image_size / feature_map_for_RoIPool.shape[1] )
            # normalization
            feature_map_mean = np.mean(feature_map_for_RoIPool)
            feature_map_std = np.std(feature_map_for_RoIPool)
            feature_map_for_RoIPool = (feature_map_for_RoIPool-feature_map_mean ) / feature_map_std

            del data

            true_bboxes = np.array([row[3]/r for row in csv_data.values if row[0] == f.replace('.jpg','')])
            if len(true_bboxes)==0:
                continue 

            true_bboxes = xywh_xyXY(true_bboxes)

            enlarged_bboxes = enlarge(true_bboxes, bbox_padding = 15)

            rois, targets, scores = produce_batch( filesDIR+f ,true_bboxes)  

            if len(rois) <= 0 :
                continue

            batch_fmaps.append(feature_map_for_RoIPool[0])
            image_counter += 1

            for i in range(len(rois)):
                crop = extract_crop(rois[i])
                batch_rois.append(crop)
                batch_inds.append(int(image_counter-1))  

                batch_scores.append(scores[i])
                batch_targets.append(targets[i])

                if (len(batch_rois)==BATCH_SIZE): 
                    all_fmaps = np.zeros((len(batch_rois),feature_map_for_RoIPool.shape[1],feature_map_for_RoIPool.shape[2],feature_map_for_RoIPool.shape[3])) 
                    # The input must share first dimension. We thus fill the batch of fmaps with zeros
                    for useful_ind in range(image_counter):
                      all_fmaps[useful_ind] = batch_fmaps[useful_ind]

                    batch_pooled_rois = tf.image.crop_and_resize( np.asarray(all_fmaps),  np.asarray(batch_rois) , np.asarray(batch_inds), (RoI_Pool_size,RoI_Pool_size))

                    if not a.any() or not b.any() or not c.any() or not d.any():
                        print("empty array found.")
                    yield batch_pooled_rois , [ np.asarray(batch_targets) ,to_categorical(np.asarray(batch_scores))]
                    batch_rois=[]
                    batch_inds=[]

                    batch_scores=[]
                    batch_targets=[]

                    batch_fmaps = []
                    image_counter = 0
                    # Scanning of rois in image continues if scan over rois hasnt ended. We need to keep the corresponding fmap!
                    if (i < len(rois)-1):
                      batch_fmaps.append(feature_map_for_RoIPool[0])
                      image_counter += 1

# Hyperparameters of the RPN
anchor_number = 9 # Number of anchors per point in the feature map
ANCHOR_RATIOS = [0.5,1,2]
ANCHOR_SCALES = np.asarray([2,4,6])

#Directories
trainDIR = 'train/'
testDIR = 'test/'

# Pooling
original_image_size = 1024
r = 1
image_size = int(round(1024/r))

# Loading CNN
model_CNN = load_model('model_checkpoint_CNN.h5')
model_CNN.load_weights('model_checkpoint_CNN.h5')
# Cutting model
number_feature_maps = 512 # this depends on the structure of the CNN. 
layer_name = 'batch_normalization_9'
CNN_model_cut = Model(inputs=model_CNN.input, outputs=model_CNN.get_layer(layer_name).output)

number_feature_maps_ROI = 256
CNN_model_RoI = Model(inputs=model_CNN.input, outputs=model_CNN.get_layer('batch_normalization_8').output)

# LLoading RPN model. fg  >=0.6 and bg <= 0.3, not excluding border anchors
RPN_model = load_model('model_checkpoint_RPN.h5',custom_objects={'loss_cls':loss_cls,'huberGTZ':huberGTZ})
RPN_model.load_weights('model_checkpoint_RPN.h5')

# Hyperparameters of the RCNN
BG_FG_FRAC_RCNN= 1
FG_THRESHOLD_RCNN = 0.5

BG_THRESH_HI = 0.2
BG_THRESH_LO = 0.0

RoI_Pool_size = 7  

#nms. 1 =  keep all overlaps
NSM_THRESHOLD = 0.7   
post_nms_N=600
N = 2000


# Loading csv data
csv_data = pd.read_csv('train.csv')
csv_data['bbox']=csv_data['bbox'].apply(lambda F: np.array([float(i) for i in  F.replace('[','').replace(']','').split(',')  ]) )

##############################
# RCNN model
##############################
input_pooled_rois = Input(batch_shape=(None,RoI_Pool_size,RoI_Pool_size,number_feature_maps_ROI))

flat1 = Flatten()(input_pooled_rois)

fc1 = Dense(
        units=1024,
        activation="relu",
        name="fc2"
    )(flat1)
fc1=BatchNormalization()(fc1)

output_deltas = Dense(
        units=4,
        activation="linear",
        kernel_initializer="uniform",
        name="deltas2"
    )(fc1)

output_scores = Dense(
        units=1 * 2,
        activation="softmax",
        kernel_initializer="uniform",
        name="scores2"
    )(fc1)

model=Model(inputs=input_pooled_rois,outputs=[output_deltas,output_scores])
model.summary()
model.compile(optimizer='adam', loss={'deltas2':huberGTZ, 'scores2':'binary_crossentropy'})

EPOCH_NUMBER = 40
STEPS = 200
BATCH_SIZE = 256
#Learning rate
def lr_schedule(epoch):
    lrate = 0.0002
    return lrate
    
##################   start train   #######################
checkpoint = ModelCheckpoint('model_checkpoint_RCNN_nopool_v11.h5', monitor='loss', verbose=1, save_best_only=True, mode='min')
history = model.fit_generator(generator = input_generator(trainDIR), steps_per_epoch=STEPS, epochs=EPOCH_NUMBER,callbacks=[checkpoint,LearningRateScheduler(lr_schedule)])
# ,validation_data = input_generator(testDIR),validation_steps=int(STEPS*0.2)

print('Done.')