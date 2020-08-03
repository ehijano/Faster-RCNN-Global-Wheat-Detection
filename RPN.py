import datetime
import os
from os import listdir
from os.path import isfile, join
import chart_studio.plotly as py
import plotly.graph_objs as go
import numpy as np
from numpy import asarray
from numpy import load
import random
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw
import pandas as pd
import matplotlib.patches as patches
#ML imports
import tensorflow as tf
import keras
from keras.losses import MSE
from keras.models import Sequential
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.datasets import cifar10
from keras import regularizers
from keras.callbacks import LearningRateScheduler
import json
from keras.models import load_model
from keras.models import model_from_json
from keras.models import Model
from keras.callbacks import LearningRateScheduler
from utils import huberGTZ, xywh_xyXY, generate_anchors, draw_anchors,bbox_overlaps,unmap,bbox_transform,loss_cls, smoothL1, bbox_transform_inv, clip_boxes, py_cpu_nms, _ratio_enum,_scale_enum
from keras.callbacks import ModelCheckpoint

# Pre-trained CNN model
model_CNN = load_model('model_CNN.h5')
model_CNN.load_weights('model_CNN.h5')


number_feature_maps = 512 # this depends on the structure of the CNN. 
layer_name = 'batch_normalization_9'
pretrained_model = Model(inputs=model_CNN.input, outputs=model_CNN.get_layer(layer_name).output)


BATCH_SIZE = 256 
FILTER_SIZE = 3
anchor_number = 3*3*1
ANCHOR_RATIOS = [0.5,1,2]
ANCHOR_SCALES = np.asarray([2,4,6])

BG_FG_FRAC = 2
FG_THRESHOLD = 0.6
BG_THRESHOLD = 0.3

trainDIR = 'train/'

# Pooling
original_image_size = 1024
r = 1
image_size = int(round(1024/r))

# Loading csv data
csv_data = pd.read_csv('train.csv')
csv_data['bbox']=csv_data['bbox'].apply(lambda F: np.array([float(i) for i in  F.replace('[','').replace(']','').split(',')  ]) )

# BATCH GENERATION
def produce_batch(image_file, true_boxes):

    image_name = image_file.replace('.jpg','').replace(trainDIR ,'')
    image = Image.open(image_file).resize((image_size ,image_size ), Image.NEAREST)
    data =  asarray(image)/255.0
    del image
    feature_map = pretrained_model.predict(data.reshape(-1,data.shape[0],data.shape[1],data.shape[2]))
    del data  

    feature_size = feature_map.shape[1]
    feature_stride = int( image_size / feature_size ) 
    number_feature_points = feature_size * feature_size 
    shift = np.arange(0, feature_size) * feature_stride
    shift_x, shift_y = np.meshgrid(shift, shift)
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel())).transpose() 
    base_anchors = generate_anchors(feature_stride, feature_stride,ratios = ANCHOR_RATIOS, scales = ANCHOR_SCALES)
    all_anchors = (base_anchors.reshape((1, anchor_number, 4)) + shifts.reshape((1, number_feature_points, 4)).transpose((1, 0, 2)))
    total_anchor_number = anchor_number*number_feature_points
    all_anchors = all_anchors.reshape((total_anchor_number , 4))

    #only keep anchors inside image+border.
    border=0 # could also be FILTER_SIZE x feature stride
    inds_inside = np.where(
            (all_anchors[:, 0] >= -border) &
            (all_anchors[:, 1] >= -border) &
            (all_anchors[:, 2] < image_size+border ) &  
            (all_anchors[:, 3] < image_size+border)    
    )[0]
    anchors=all_anchors[inds_inside]
    useful_anchor_number = len(inds_inside)


    overlaps = bbox_overlaps(anchors, true_boxes) 

    which_box = overlaps.argmax(axis=1) # Which true box has more overlap with each anchor?
    anchor_max_overlaps = overlaps[np.arange(overlaps.shape[0]), which_box] 

    which_anchor = overlaps.argmax(axis=0) # Which anchor has more overlap for each true box?
    box_max_overlaps = overlaps[which_anchor, np.arange(overlaps.shape[1])] 
    which_anchor_v2 = np.where(overlaps == box_max_overlaps)[0]

    labels = np.empty((useful_anchor_number, ), dtype=np.float32)
    labels.fill(-1)

    labels[ which_anchor_v2 ] = 1
    labels[ anchor_max_overlaps >= FG_THRESHOLD] = 1
    labels[ anchor_max_overlaps <= BG_THRESHOLD] = 0

    fg_inds = np.where(labels == 1)[0]
    bg_inds = np.where(labels == 0)[0]

    num_fg = int(BATCH_SIZE/(1+BG_FG_FRAC))
    if len(fg_inds) > num_fg:
      disable_inds = np.random.choice(fg_inds, size=(len(fg_inds) - num_fg), replace=False)
      labels[disable_inds] = -1
    fg_inds = np.where(labels == 1)[0]

    num_bg = int(len(fg_inds) * BG_FG_FRAC) 
    if len(bg_inds) > num_bg:
        disable_inds = np.random.choice(bg_inds, size=(len(bg_inds) - num_bg), replace=False)
        labels[disable_inds] = -1
    bg_inds = np.where(labels == 0)[0]

    anchor_batch_inds = inds_inside[labels!=-1]
    np.random.shuffle(anchor_batch_inds)  
    feature_batch_inds=(anchor_batch_inds / anchor_number).astype(np.int)

    pad_size = int((FILTER_SIZE-1)/2)
    padded_fcmap=np.pad(feature_map,((0,0),(pad_size,pad_size),(pad_size,pad_size),(0,0)),mode='constant')
    padded_fcmap=np.squeeze(padded_fcmap)
    batch_tiles=[]  
    for ind in feature_batch_inds:
        # x,y are the point in the feature map pointed at by feature_batch_inds indices
        x = ind % feature_size
        y = int(ind/feature_size)
        fc_snip=padded_fcmap[y:y+FILTER_SIZE,x:x+FILTER_SIZE,:] 
        batch_tiles.append(fc_snip)

    # unmap creates another array of labels that includes a -1 for the originally deleted anchors for being out of bounds.
    full_labels = unmap(labels, total_anchor_number , inds_inside, fill=-1)
    batch_labels =full_labels.reshape(-1,1,1,1*anchor_number)[feature_batch_inds]


    targets = np.zeros((len(inds_inside), 4), dtype=np.float32)
    pos_anchors=all_anchors[inds_inside[labels==1]] # positive anchors
    targets = bbox_transform(pos_anchors, true_boxes[which_box, :][labels==1])
    targets = unmap(targets, total_anchor_number, inds_inside[labels==1], fill=0)
    batch_targets = targets.reshape(-1,1,1,4*anchor_number)[feature_batch_inds]

    return np.asarray(batch_tiles), batch_labels.tolist(), batch_targets.tolist()

def input_generator():
    batch_tiles=[]
    batch_labels=[]
    batch_bboxes=[]
    count=0
    while 1:
        print('# of files analyzed: ',count)
        for f in listdir(trainDIR):
            count += 1
            true_bboxes = np.array([row[3]/r for row in csv_data.values if row[0] == f.replace('.jpg','')])
            #print('# of true boxes in ',f,': ',len(true_bboxes_aux))
            if len(true_bboxes)==0:
                continue #scans next file instead of producing any batch
            #print('\n max w: ',np.max(true_bboxes_aux[:,2]),'max h: ',np.max(true_bboxes_aux[:,3]))
            #print('\n min w: ',np.min(true_bboxes_aux[:,2]),'min h: ',np.min(true_bboxes_aux[:,3]))
            # We have xmin y min, width, height. We need to have xmin, ymin, xmax, ymax (this is how utils is coded)

            true_bboxes = xywh_xyXY(true_bboxes)

            true_bboxes = enlarge(true_bboxes,bbox_padding = 15)

            tiles, labels, bboxes = produce_batch(trainDIR+f, true_bboxes ) #bboxes are displacements of xmin,ymin,w,h
            #print('looking at f: ',f, ' it has this # of tiles: ',len(tiles))
            for i in range(len(tiles)):
                batch_tiles.append(tiles[i])
                batch_labels.append(labels[i])
                batch_bboxes.append(bboxes[i])
                if(len(batch_tiles)==BATCH_SIZE): #Once we fill a batch size, we yield the result and clear the data. Once the function is called again, it continues the loop.
                    a=np.asarray(batch_tiles)
                    b=np.asarray(batch_labels)
                    c=np.asarray(batch_bboxes)
                    if not a.any() or not b.any() or not c.any():
                        print("empty array found.")
                    yield a, [b, c]
                    batch_tiles=[]
                    batch_labels=[]
                    batch_bboxes=[]

##############################
# RPN MODEL
##############################
feature_map_tile = keras.Input(shape=(FILTER_SIZE,FILTER_SIZE,number_feature_maps))

convolution = Conv2D(
    filters=512,
    kernel_size=(FILTER_SIZE, FILTER_SIZE),
    name="fmapconvolution"
)(feature_map_tile)

output_deltas = Conv2D(
    filters = int(4 * anchor_number),
    kernel_size=(1, 1),
    activation="linear",
    kernel_initializer="uniform",
    name="deltas1"
)(convolution)

output_scores = Conv2D(
    filters = int(1 * anchor_number),
    kernel_size=(1, 1),
    activation="sigmoid",
    kernel_initializer="uniform",
    name="scores1"
)(convolution)

model = Model(inputs=[feature_map_tile], outputs=[output_scores, output_deltas])
model.compile(optimizer='adam', loss={'scores1':loss_cls, 'deltas1':huberGTZ})

##### training #####
EPOCH_NUMBER = 20
STEPS = 300

#Learning rate
def lr_schedule(epoch):
    lrate = 0.001*(1-0.9*epoch/EPOCH_NUMBER)
    return lrate

# Fit
checkpoint = ModelCheckpoint('model_checkpoint_RPN.h5', monitor='loss', verbose=1, save_best_only=True, mode='min')
history = model.fit_generator(input_generator(), steps_per_epoch=STEPS, epochs=EPOCH_NUMBER, verbose=1,callbacks=[checkpoint,LearningRateScheduler(lr_schedule)])

# Save to disk
model_json = model.to_json()
with open('model_RP.json', 'w') as json_file:
    json_file.write(model_json)
model.save_weights('model_RPN.h5') 

print('Done.')