import os
from PIL import Image, ImageDraw
from numpy import asarray
from matplotlib import pyplot as plt
import numpy as np
import csv
import pandas as pd
from os import listdir
from os.path import isfile, join
import itertools
import random
from numpy import save

# no intersect function
def no_intersect(c_X,c_Y,bboxes):
    center_X = c_X*target_size+target_size/2
    center_Y = c_Y*target_size+target_size/2
    min_distance=image_size**2
    for bbox in bboxes:
        center_box_X = bbox[0]+round(bbox[2]/2)
        center_box_Y = bbox[1]+round(bbox[3]/2)
        box_distance = (center_box_X-center_X)**2 + (center_box_Y-center_Y)**2
        if box_distance < min_distance:
            min_distance = box_distance
    if min_distance < 2*(target_size)**2:
        return False
    return True


targetDIR = 'batches/'
trainDIR = 'train/'

# Loading train.csv
csv_data = pd.read_csv('train.csv')
csv_data['bbox']=csv_data['bbox'].apply(lambda F: np.array([float(i)/r for i in  F.replace('[','').replace(']','').split(',')  ]) )
# Reading training files
training_files = [f for f in listdir(trainDIR) if isfile(join(trainDIR , f))]
number_images = len(training_files) 

# Resize image if needed
original_image_size = 1024
r=1
image_size = round(original_image_size/r)
target_size = round(256/r)

# Initialize training data
x_full = []
y_full = []
x_empty = []
y_empty = []

for image_name in training_files:
    image = Image.open("train\\"+image_name).resize((image_size ,image_size ), Image.NEAREST)
    data =  asarray(image)
    # The bounding boxes are entry number 3 in each row in the csv file. The name is entry 0
    bboxes = [row[3] for row in csv_data.values if row[0] == image_name.replace('.jpg','')]
    # Generate snips with spikes
    for bbox in bboxes:
        snip_X_min = round(bbox[0]+round(bbox[2]/2)-target_size/2)
        snip_Y_min = round(bbox[1]+round(bbox[3]/2)-target_size/2)
        if (snip_Y_min>=0) and (snip_Y_min+target_size<=image_size ) and (snip_X_min>=0) and (snip_X_min+target_size<=image_size ) and (bbox[2]*bbox[3]>3500/(r*r) ):
            snip = data[snip_Y_min:snip_Y_min+target_size,snip_X_min:snip_X_min+target_size,:]
            x_full.append(snip)
            y_full.append(1)
            del snip
    # Generate snips with background
    for c_X, c_Y in itertools.product(range(round(image_size/target_size)),range(round(image_size/target_size))):
        if no_intersect(c_X,c_Y,bboxes):
            snip_X_min = c_X*target_size
            snip_Y_min = c_Y*target_size
            snip = data[snip_Y_min:snip_Y_min+target_size,snip_X_min:snip_X_min+target_size,:]
            x_empty.append(snip)
            y_empty.append(0)
            del snip

print('# of empty samples: ',len(y_empty))
print('# of full samples: ',len(y_full))
size = min(12000, len(y_full), len(y_empty))
print('# of samples per class: ', size)

# Resize
x_full, y_full = zip(*random.sample(list(zip(x_full, y_full)), size ))
x_empty, y_empty = zip(*random.sample(list(zip(x_empty, y_empty)), size ))

x_train_list = itertools.chain(x_full, x_empty)
y_train_list = itertools.chain(y_full, y_empty)

x_train_list, y_train_list = zip(*random.sample(list(zip(x_train_list, y_train_list)),  int(2*size)  ))

print('Saving...')

# Define parameters to split between train and test data
data_size = len(x_train_list)
train_split = int(round(0.8*data_size))

# Store train data and their file names
file_names_train = []
y_train = []
for counter in range(0,train_split):
    file_name = targetDIR+'data_'+str(counter)+'.npy'

    file_names_train.append(file_name)
    y_train.append(y_train_list[counter])

    save(file_name, x_train_list[counter])

file_names_test = []
y_test = []
for counter in range(train_split+1,data_size-1):
    file_name = targetDIR+'data_'+str(counter)+'.npy'

    file_names_test.append(file_name)
    y_test.append(y_train_list[counter])

    save(file_name, x_train_list[counter])

save(targetDIR+'y_train.npy',y_train)
save(targetDIR+'y_test.npy',y_test)
save(targetDIR+'file_names_train.npy',file_names_train)
save(targetDIR+'file_names_test.npy',file_names_test)

print('Done.')