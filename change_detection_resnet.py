# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 11:48:11 2019

@author: Angelo Antonio Manzatto

@references:
    
This project was made using as reference Change Detection Net (CDNET):
Link: http://jacarini.dinf.usherbrooke.ca/
 
Article of Reference:
Y. Wang, P.-M. Jodoin, F. Porikli, J. Konrad, Y. Benezeth, and P. Ishwar, CDnet 2014: 
An Expanded Change Detection Benchmark Dataset, in Proc. IEEE Workshop on Change Detection (CDW-2014) at CVPR-2014, pp. 387-394. 2014

"""

##################################################################################
# Libraries
##################################################################################  
import os
import csv
import requests
import io
import glob
import zipfile

import numpy as np

import matplotlib.pyplot as plt

import cv2

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from keras.models import Model

from keras.layers import Reshape, Lambda, Input,Activation
from keras.layers import Conv2D,  MaxPooling2D, ZeroPadding2D, BatchNormalization, Conv2DTranspose, Add
from keras.layers import AveragePooling3D, MaxPooling3D

from keras.callbacks import ModelCheckpoint, CSVLogger

##################################################################################
# Download Dataset
##################################################################################  

dataset_folder = 'dataset'
database_name = 'turnpike_0_5fps'

# Dataset Link
url = 'http://jacarini.dinf.usherbrooke.ca/static/dataset/lowFramerate/turnpike_0_5fps.zip'   

# Create Dataset folder 
if not os.path.exists(dataset_folder):
    os.makedirs(dataset_folder)

# Donwload dataset if doesn't exist
if not os.path.exists(os.path.join(dataset_folder,database_name)):
    
    # Request files
    r = requests.get(url, allow_redirects=True)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    
    # Extract data to folder
    z.extractall(dataset_folder)

############################################################################################
# Files and folders
############################################################################################
    
# Folders for input images and ground truth images    
input_folder = os.path.join(dataset_folder,database_name,'input')
gt_folder = os.path.join(dataset_folder,database_name,'groundtruth')

# This file defines the index of the images with and withoud a ground truth
temporalROI_file = os.path.join(dataset_folder,database_name,'temporalROI.txt')

f = open(temporalROI_file, "r")
background_idx, roi_idx = f.readline().split()

# Get indexes between just the background images provided for static model and images where we have a ground truth annotation
background_idx = int(background_idx)
roi_idx = int(roi_idx)

# We have just 350 images labeled
roi_images = 350
f.close()

# ROI Masks
roi_file = os.path.join(dataset_folder,database_name,'ROI.jpg')
roi_mask_file = os.path.join(dataset_folder,database_name,'ROI.bmp')

roi_image = plt.imread(roi_file)
roi_mask = plt.imread(roi_mask_file)

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8))
f.tight_layout()
ax1.imshow(roi_image)
ax1.set_title('Region Of Interest (ROI) ', fontsize=10)
ax2.imshow(roi_mask, cmap = 'gray')
ax2.set_title('Region Of Interest (ROI) Mask', fontsize=10)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

############################################################################################
# Load data
############################################################################################

# image files
input_image_files = glob.glob(os.path.join(input_folder,'*.jpg'))  
gt_image_files = glob.glob(os.path.join(gt_folder,'*.png'))   

# Sort names to avoid connecting wrong numbered files
input_image_files.sort()
gt_image_files.sort()

assert(len(input_image_files) == len(gt_image_files))

# Our dataset is created using the tuple (input image file, ground truth image file)
data = []
for input_image, gt_image in zip(input_image_files,gt_image_files):
    
    data.append((input_image,gt_image))

# This data will be used to create the background model to train the NN
background_data = data[:background_idx-1]

# This data will be used to train / test the model
train_test_data = data[background_idx-1:background_idx-1+roi_images]

# This data will be used to validate the model
valid_data = data[background_idx-1+roi_images:]

# Plot some samples from training set
n_samples = 5

for i in range(n_samples):
    
    # define the size of images
    f, (ax1, ax2) = plt.subplots(1, 2)
    f.set_figwidth(12)
    
    # randomly select a sample
    idx = np.random.randint(0, len(train_test_data))
    input_image_path, gt_image_path = train_test_data[idx]

    input_image = plt.imread(input_image_path)
    gt_image = plt.imread(gt_image_path)
    
    ax1.imshow(input_image)
    ax1.set_title('Input Image # {0}'.format(idx))
    
    ax2.imshow(gt_image)
    ax2.set_title('Ground Truth Image # {0}'.format(idx))
        
############################################################################################
# Build a background model from a batch of images. We use this to create the static image 
# that will be used to help training the model taking the average value from all files
############################################################################################
def build_background_model(background_data):
    
    n_samples = len(background_data)
    
    bg_image_path, gt_image_path = background_data[0]
    
    first_sample = plt.imread(bg_image_path)
    
    background_model = np.zeros_like(first_sample).astype('float32')
    
    for i in range(n_samples):
        
        bg_image_path, gt_image_path = background_data[i]
        
        sample = plt.imread(bg_image_path)

        background_model += sample
    
    background_model /= n_samples
    
    return background_model.astype('uint8') # Clip to integers of type uint8    

############################################################################################
# Data Augmentation 
############################################################################################
    
#############################
# Resize Image
#############################
class Resize(object):
    
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, image):
        
        # Get image shape
        h, w = image.shape[:2]
        
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size
            
        new_h, new_w = int(new_h), int(new_w)

        image = cv2.resize(image, (new_w, new_h))
      
        return image

#############################
# Remove camera data from Image
#############################
class MaskROI(object):
    
    def __init__(self, roi_mask):
        
        mask = roi_mask > 0

        # Coordinates of non-black pixels.
        coords = np.argwhere(mask)
        
        x0, y0 = coords.min(axis=0)
        x1, y1 = coords.max(axis=0)  

        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1+1

    def __call__(self, image):
        
        cropped_image = image[self.x0:self.x1, self.y0:self.y1]
        
        return cropped_image
   
############################################################################################
# Test Data Augmentation 
############################################################################################
def plot_transformation(transformation, n_samples = 3):

    for i in range(n_samples):
    
        # define the size of images
        f, (ax1, ax2) = plt.subplots(1, 2)
        f.set_figwidth(14)

        # randomly select a sample
        idx = np.random.randint(0, len(train_test_data))
        input_image_path, gt_image_path = train_test_data[idx]

        input_image = plt.imread(input_image_path)
        gt_image = plt.imread(gt_image_path)

        new_input_image = transformation(input_image)
        new_gt_image = transformation(gt_image)
        
        if(new_input_image.shape[-1] == 1):
            ax1.imshow(np.squeeze(new_input_image), cmap = 'gray')
        else:
            ax1.imshow(new_input_image)
        ax1.set_title('Original')

        if(new_gt_image.shape[-1] == 1):
            ax2.imshow(np.squeeze(new_gt_image), cmap = 'gray')
        else:
            ax2.imshow(new_gt_image)
        ax2.set_title(type(transformation).__name__)

        plt.show()
        
##########################
# Resize Test
##########################
resize = Resize((224, 224))
plot_transformation(resize)

##########################
# ROI Mask Test
##########################
mask = MaskROI(roi_mask)
plot_transformation(mask)

############################################################################################
# Create Dataset
############################################################################################
    
# Create X, y tuple from image_path, key_pts tuple
def createXy(data, background_image, transformations = None):
    
    input_image_path, gt_image_path = data
    
    input_image = plt.imread(input_image_path)
    gt_image = plt.imread(gt_image_path)
    
    # Apply transformations for the tuple (image, labels, boxes)
    if transformations:
        for t in transformations:
            input_image = t(input_image)
            gt_image  = t(gt_image)
            background_image = t(background_image) 
            
    stacked_image = np.uint8(np.concatenate([input_image,background_image],2))
    
    # Get only one channel since the GT is already in gray scale but with three channels
    gt_image = gt_image[:,:,0]
    gt_image = np.expand_dims(gt_image, axis = -1)         
    
    return stacked_image,gt_image

# Generator for using with model
def generator(data, background_image,  transformations = None, batch_size = 4, shuffle_data= True):
    
    n_samples = len(data)
    
    # Loop forever for the generator
    while 1:
        
        if shuffle_data:
            data = shuffle(data)
        
        for offset in range(0, n_samples, batch_size): 
            
            batch_samples = data[offset:offset + batch_size]
            
            X = []
            y = []
            
            for sample_data in batch_samples:
                
                image, target = createXy(sample_data, background_image, transformations)

                X.append(image)
                y.append(target)
                
            X = np.asarray(X).astype('float32')
            y = np.asarray(y).astype('float32')
            
            yield (shuffle(X, y))

############################################################################################
# ResNet Convolution Block
############################################################################################
def conv_block(inputs, kernel_size, filters, strides, block_id):
    
    f1, f2, f3 = filters
    
    x = Conv2D(f1, kernel_size=(1,1), use_bias=False, kernel_initializer='he_normal', name='block_' + str(block_id) + '_conv_conv2d_1')(inputs)
    x = BatchNormalization(name='block_' + str(block_id) + '_conv_batch_1')(x)
    x = Activation('relu', name='block_' + str(block_id) + '_expand_relu')(x)
    
    x = Conv2D(f2, kernel_size=kernel_size, strides=strides, padding='same', use_bias=False, kernel_initializer='he_normal', name='block_' + str(block_id) + '_conv_conv2d_2')(x)
    x = BatchNormalization(name='block_' + str(block_id) + '_conv_batch_2')(x)
    x = Activation('relu', name='block_' + str(block_id) + '_depthwise_relu')(x)
    
    x = Conv2D(f3, kernel_size=(1,1), use_bias=False, kernel_initializer='he_normal', name='block_' + str(block_id) + '_project_conv2d')(x)
    x = BatchNormalization(name='block_' + str(block_id) + '_project_batch')(x)
    
    shortcut = Conv2D(f3, kernel_size=(1,1), strides=strides, use_bias=False, kernel_initializer='he_normal', name='block_' + str(block_id) + '_shortcut_conv2d')(inputs)
    shortcut = BatchNormalization(name='block_' + str(block_id) + '_shortcut_batch')(x)
    
    x = Add(name='block_' + str(block_id) + '_add')([x, shortcut])
    x = Activation('relu',name='block_' + str(block_id) + '_add_relu')(x)
    
    return x

############################################################################################
# ResNet Identity Block
############################################################################################
def identity_block(inputs,kernel_size, filters, block_id):
    
    f1, f2, f3 = filters
    
    x = Conv2D(f1, kernel_size=(1,1), use_bias=False, kernel_initializer='he_normal', name='block_' + str(block_id) + '_identity_conv2d_1')(inputs)
    x = BatchNormalization(name='block_' + str(block_id) + '_identity_batch_1')(x)
    x = Activation('relu', name='block_' + str(block_id) + '_identity_relu_1')(x)
    
    x = Conv2D(f2, kernel_size = kernel_size, padding='same', use_bias=False, kernel_initializer='he_normal', name='block_' + str(block_id) + '_identity_conv2d_2')(x)
    x = BatchNormalization(name='block_' + str(block_id) + '_identity_batch_2')(x)
    x = Activation('relu',name='block_' + str(block_id) + '_identity_relu_2')(x)
    
    x = Conv2D(f3, kernel_size=(1,1), use_bias=False, kernel_initializer='he_normal', name='block_' + str(block_id) + '_identity_conv2d_3')(x)
    x = BatchNormalization(name='block_' + str(block_id) + '_identity_batch_3')(x)
    
    x = Add(name='block_' + str(block_id) + '_add')([x, inputs])
    x = Activation('relu', name='block_' + str(block_id) + '_identity_relu_3')(x)
    
    return x

def ChangeDetectionNet(input_shape=(224,224,6)):
    
    Image = Input(shape=input_shape)
    Normalize = Lambda(lambda x: x / 255.0 - 0.5)(Image)
    
    ################################
    # Adapter Block to convert from 6 channels to 3
    ################################
    
    encoder = Conv2D(3, kernel_size=(1,1), padding='same', kernel_initializer='he_normal', name='pre_conv')(Normalize)
    
    ################################
    # Block 1
    ################################
    
    encoder = Conv2D(64, kernel_size=(7,7),strides=(2,2), padding='same', kernel_initializer='he_normal', name = 'conv1')(encoder)
    encoder = BatchNormalization(name = 'batch_1')(encoder)
    encoder = Activation('relu',name='relu_1')(encoder)
    encoder = ZeroPadding2D(padding=(1,1), name='zero_pad_1')(encoder)
    encoder = MaxPooling2D(pool_size=(3,3), strides=(2,2), name='max_pool_1')(encoder)
    
    ################################
    # Block 1
    ################################
    encoder = conv_block(encoder, kernel_size=(3,3), filters = [64, 64, 256], strides=(1,1), block_id = 0)
    encoder = identity_block(encoder,kernel_size=(3,3), filters= [64, 64, 256], block_id = 1)
    encoder = identity_block(encoder,kernel_size=(3,3), filters= [64, 64, 256], block_id = 2)
    
    ################################
    # Block 2
    ################################
    encoder = conv_block(encoder, kernel_size=(3,3), filters = [128, 128, 512], strides=(2,2), block_id = 3)
    encoder = identity_block(encoder,kernel_size=(3,3), filters= [128, 128, 512], block_id = 4)
    encoder = identity_block(encoder,kernel_size=(3,3), filters= [128, 128, 512], block_id = 5)
    encoder = identity_block(encoder,kernel_size=(3,3), filters= [128, 128, 512], block_id = 6)
    
    ################################
    # Block 3
    ################################
    encoder = conv_block(encoder, kernel_size=(3,3), filters = [256, 256, 1024], strides=(2,2), block_id = 7)
    encoder = identity_block(encoder,kernel_size=(3,3), filters= [256, 256, 1024], block_id = 8)
    encoder = identity_block(encoder,kernel_size=(3,3), filters= [256, 256, 1024], block_id = 9)
    encoder = identity_block(encoder,kernel_size=(3,3), filters= [256, 256, 1024], block_id = 10)
    encoder = identity_block(encoder,kernel_size=(3,3), filters= [256, 256, 1024], block_id = 11)
    encoder = identity_block(encoder,kernel_size=(3,3), filters= [256, 256, 1024], block_id = 12)
    
    ################################
    # Block 4
    ################################
    encoder = conv_block(encoder, kernel_size = (3,3), filters = [512, 512, 2048], strides=(2,2), block_id = 13)
    encoder = identity_block(encoder,kernel_size= (3,3), filters= [512, 512, 2048], block_id = 14)
    encoder = identity_block(encoder,kernel_size= (3,3), filters= [512, 512, 2048], block_id = 15)
    
    ################################
    # Decode Block 1
    ################################
    
    decoder = Conv2DTranspose(2048,kernel_size=(1,1),strides=(2,2))(encoder)
    decoder = Reshape((14,14,2048,1))(decoder)
    
    decoder = AveragePooling3D(pool_size=(1,1,48),strides=(1,1,40))(decoder)

    decoder = Reshape((14,14,51))(decoder)
    decoder = Conv2DTranspose(32, kernel_size=(3,3),strides=(2,2))(decoder)
    
    decoder = Reshape((29,29,32,1))(decoder)
    decoder = MaxPooling3D(pool_size=(2,2,2),strides=(1,1,2))(decoder)

    decoder = Reshape((28,28,16))(decoder)
    
    # Padding is an important step here unless you want an horrible border effect on the prediction
    decoder = ZeroPadding2D((1,1))(decoder)
    
    decoder = Conv2DTranspose(8, kernel_size=(2,2),strides=(2,2))(decoder)
    decoder = MaxPooling2D(pool_size=(3,3),strides=(1,1))(decoder)

    decoder = Conv2DTranspose(4, kernel_size=(2,2),strides=(2,2))(decoder)
    decoder = MaxPooling2D(pool_size=(3,3),strides=(1,1))(decoder)
    
    decoder = Conv2DTranspose(1, kernel_size=(2,2),strides=(2,2))(decoder)
    decoder = MaxPooling2D(pool_size=(3,3),strides=(1,1))(decoder)
    
    decoder = Conv2D(1,kernel_size=(1,1),strides=(1,1), activation='sigmoid')(decoder)
    decoder = MaxPooling2D(pool_size=(3,3),strides=(1,1))(decoder)
    
    model = Model(inputs=Image,outputs=decoder)

    return model

############################################################################################
# Training pipeline
############################################################################################

# Data augmentation 
train_transformations = [
        MaskROI(roi_mask),
        Resize((224,224))
        ]

test_transformations = [
        MaskROI(roi_mask),
        Resize((224,224))
        ]

valid_transformations = [
        MaskROI(roi_mask),
        Resize((224,224))
        ]

# Hyperparameters
epochs = 200
batch_size = 4 # Change this value if you have more GPU Power
learning_rate = 0.001
weight_decay = 5e-4
momentum = .9

train_data, test_data = train_test_split(train_test_data, test_size=0.20, random_state=42)

background_image = build_background_model(background_data)

train_generator = generator(train_data, background_image, train_transformations, batch_size)
test_generator = generator(test_data, background_image, test_transformations, batch_size)

# callbacks
model_path = 'saved_models'

# File were the best model will be saved during checkpoint     
model_file = os.path.join(model_path,'change_detection-{val_loss:.4f}.h5')

# Check point for saving the best model
check_pointer = ModelCheckpoint(model_file, monitor='val_loss', mode='min',verbose=1, save_best_only=True)

# Logger to store loss on a csv file
csv_logger = CSVLogger(filename='change_detection.csv',separator=',', append=True)

model = ChangeDetectionNet()
model.compile(optimizer='adam',loss='binary_crossentropy', metrics = ['accuracy'])

model.summary()

history = model.fit_generator(train_generator,steps_per_epoch=int(len(train_data) / batch_size),
                              validation_data=test_generator,validation_steps=int(len(test_data) / batch_size),
                              epochs=epochs, verbose=1, callbacks=[check_pointer,csv_logger],workers=1)

############################################################################################
# Predict
############################################################################################

# If we want to test on a pre trained model use the following line
# model.load_weights(os.path.join(model_path,'<path to model>'), by_name=False)

n_samples = 5

for i in range(n_samples):
    
    f, (ax1, ax2, ax3) = plt.subplots(1, 3)
    f.set_figwidth(14)
    
    # randomly select a sample
    idx = np.random.randint(0, len(valid_data))
    sample = valid_data[idx]
    
    # Read input file
    input_image_path, gt_image_path = sample
    input_image = plt.imread(input_image_path)
    
    input_image_copy = np.copy(input_image)
    background_image_copy = np.copy(background_image)

    # Apply transformations to input and background image
    for t in valid_transformations:
        input_image_copy = t(input_image_copy)
        background_image_copy = t(background_image_copy)
        
    # Stack images to compose the input for model [H, W, 6]
    stacked_image = np.uint8(np.concatenate([input_image_copy,background_image_copy],2))

    # Predict 
    mask_pred = model.predict(stacked_image[np.newaxis,...]) 
    
    # Squeeze
    mask_pred = np.squeeze(mask_pred[0])
    
    # Plot 
    ax1.imshow(input_image_copy.astype('uint8'))
    
    ax2.imshow(mask_pred, cmap='gray')
    
    mask_pred[mask_pred > 0.5] = 1.0
    mask_pred[mask_pred <= 0.5] = 0.0
    
    combined_image = np.copy(input_image_copy)
    combined_image[mask_pred == 0] = 0
    
    ax3.imshow(combined_image)
    
############################################################################################
# Evaluate our model
############################################################################################

mask = MaskROI(roi_mask)

measures = np.zeros((len(train_test_data),11))
statistics = []
    
for i, sample in enumerate(train_test_data):
    
    # Retrieve tuple
    input_image_path, gt_image_path = sample
    
    print("input: {0} , gt: {1}".format(input_image_path.split("\\")[-1],gt_image_path.split("\\")[-1]))
    
    # Read input file
    input_image = plt.imread(input_image_path)
    gt_image = plt.imread(gt_image_path)
    
    # Make a copy for safety
    input_image_copy = np.copy(input_image)
    background_image_copy = np.copy(background_image)
    gt_image_copy = np.copy(gt_image)
    
    # Apply transformations to input and background image. This time we will not use the ground truth image 
    # in the transformation step since we want to evaluate against the original data excluding the mask
    for t in valid_transformations:
        input_image_copy = t(input_image_copy)
        background_image_copy = t(background_image_copy)

    # Just remove the lower border of camera data
    y_true = mask(gt_image_copy)
    y_true = y_true[:,:,0]
    
    # Stack images to compose the input for model [H, W, 6]
    X = np.uint8(np.concatenate([input_image_copy,background_image_copy],2))
    
    # Predict the mask using the model
    y_pred = model.predict(stacked_image[np.newaxis,...]) 
    
    # Remove sinle dimensions
    y_pred = np.squeeze(y_pred)
    
    # Resize output from model to original size from the ground truth
    w,h = y_true.shape[:2]
    resize = Resize((w,h))
    y_pred = resize(y_pred)
    
    # Apply threshold value where we consider the predicted pixel as 1 and 0
    y_pred[y_pred > 0.5] = 1.0
    y_pred[y_pred <= 0.5] = 0.0
    
    y_true[y_true > 0.5] = 1.0
    y_true[y_true <= 0.5] = 0.0
        
    # Calculate True Positive, True Negative, False Positive, False Negative
    TP = np.sum(np.logical_and(y_pred == 1, y_true == 1))
    TN = np.sum(np.logical_and(y_pred == 0, y_true == 0))
    FP = np.sum(np.logical_and(y_pred == 1, y_true == 0))
    FN = np.sum(np.logical_and(y_pred == 0, y_true == 1))

    # Calculate Recall
    recall = TP / (TP + FN)
    
    # Calculate Specificity
    specificity =  TN / (TN + FP)
    
    # Calculate False Positive Rate
    FPR =  FP / (FP + TN)
    
    # Calculate False Negative Rate
    FNR = FN / (TP + FN)
    
    # Calculate Percentage of Wrong Classifications
    PWC =  100 * (FN + FP) / (TP + FN + FP + TN)

    # Calculate Precision
    precision =  TP / (TP + FP)
    
    # Calculate f measure
    f_measure = (2 * precision * recall) / (precision + recall)
    
    stats = { 'input_image': input_image_path,
              'gt_image' : gt_image_path,
              'TP' : TP,
              'TN' : TN,
              'FP' : FP,
              'FN' : FN,
              'recall' : recall,
              'specificity' : specificity,
              'FPR' : FPR,
              'FNR' : FNR,
              'PWC' : PWC,
              'precision' : precision,
              'f_measure' : f_measure
            }
    
    statistics.append(stats)
    
    measures[i][0] = TP
    measures[i][1] = TN
    measures[i][2] = FP
    measures[i][3] = FN
    measures[i][4] = recall
    measures[i][5] = specificity
    measures[i][6] = FPR
    measures[i][7] = FNR
    measures[i][8] = PWC
    measures[i][9] = precision
    measures[i][10] = f_measure

# Replace NaN by zeros
measures[np.isnan(measures)] = 0

# Calculate Mean Average for all our statistics
mean_score = measures.mean(axis=0)
max_score = measures.max(axis=0)
min_score = measures.min(axis=0)

# Print Average Statistics
print(50*'-')
print('Recal: Avg:{0:.2f} , Max: {1:.2f} , Min: {2:.2f}'.format(mean_score[4],max_score[4],min_score[4]))
print(50*'-')
print('Specificity: Avg:{0:.2f} , Max: {1:.2f} , Min: {2:.2f}'.format(mean_score[5],max_score[5],min_score[5]))
print(50*'-')
print('False Positive Rate: Avg:{0:.2f} , Max: {1:.2f} , Min: {2:.2f}'.format(mean_score[6],max_score[6],min_score[6]))
print(50*'-')
print('Calculate False Negative Rate: Avg:{0:.2f} , Max: {1:.2f} , Min: {2:.2f}'.format(mean_score[7],max_score[7],min_score[7]))
print(50*'-')
print('Percentage of Wrong Classifications: Avg:{0:.2f} , Max: {1:.2f} , Min: {2:.2f}'.format(mean_score[8],max_score[8],min_score[8]))
print(50*'-')
print('Precision: Avg:{0:.2f} , Max: {1:.2f} , Min: {2:.2f}'.format(mean_score[9],max_score[9],min_score[9]))
print(50*'-')
print('F-Measure: Avg:{0:.2f} , Max: {1:.2f} , Min: {2:.2f}'.format(mean_score[10],max_score[10],min_score[10]))
print(50*'-')

# Create Confusion Matrix
cm = np.zeros((2,2))
cm[0][0] = mean_score[0]
cm[0][1] = mean_score[2]
cm[1][0] = mean_score[3]
cm[1][1] = mean_score[1]

print(50*'-')
print('Confusion Matrix')
plt.matshow(cm)
plt.colorbar()
print(50*'-')

# Save results on a file
results_folder = 'results'

# Create Dataset folder 
if not os.path.exists(results_folder):
    os.makedirs(results_folder)
    
csv_columns = ['input_image','gt_image','TP','TN','FP','FN','recall','specificity','FPR','FNR','PWC','precision','f_measure']

csv_file = os.path.join(results_folder,database_name + '_stats.csv')

try:
    with open(csv_file, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns, lineterminator = '\n')
        writer.writeheader()
        for data in statistics:
            writer.writerow(data)
except IOError:
    print("I/O error") 
    
############################################################################################
# Video Testing
############################################################################################    
from moviepy.editor import ImageClip, concatenate

video_folder = 'videos'   

clips = []

for sample in valid_data:
    
    input_image_path, gt_image_path = sample
    
    # Read input file
    input_image_path, gt_image_path = sample
    input_image = plt.imread(input_image_path)
    
    input_image_copy = np.copy(input_image)
    background_image_copy = np.copy(background_image)

    # Apply transformations to input and background image
    for t in valid_transformations:
        input_image_copy = t(input_image_copy)
        background_image_copy = t(background_image_copy)
        
    # Stack images to compose the input for model [H, W, 6]
    stacked_image = np.uint8(np.concatenate([input_image_copy,background_image_copy],2))

    # Predict 
    mask_pred = model.predict(stacked_image[np.newaxis,...]) 
    
    # Squeeze
    mask_pred = np.squeeze(mask_pred[0])
    mask_pred[mask_pred > 0.5] = 1.0
    mask_pred[mask_pred <= 0.5] = 0.0
    
    combined_image = np.copy(input_image_copy)
    combined_image[mask_pred == 0] = 0
        
    combined_image = cv2.resize(combined_image,(320 ,320))
    resized_input_image = cv2.resize(input_image_copy,(320 ,320))
    
    stacked_image = np.hstack((resized_input_image, combined_image))
        
    clips.append(ImageClip(stacked_image).set_duration(1))

video = concatenate(clips, method="compose")
video.write_videofile(os.path.join(video_folder,'test_video.mp4'), fps=24)
