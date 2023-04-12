import tensorflow as tf
#Necessary Packages
import cv2     # for capturing videos
import math   # for mathematical operations
import matplotlib.pyplot as plt    # for plotting the images
%matplotlib inline
import pandas as pd
# from keras.preprocessing import image   # for preprocessing the images
import keras.utils as image
import numpy as np    # for mathematical operations
from keras.utils import np_utils, to_categorical
from skimage.transform import resize   # for resizing images
import os
import tqdm

import keras

from tqdm import tqdm,tqdm_pandas
from keras.models import Sequential
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras import applications, models, losses,optimizers
# from keras.layers.normalization import BatchNormalization. # NOTE: Importing this below
from keras.layers.pooling import MaxPooling2D, AveragePooling2D

# from keras.layers.merge import Concatenate  # NOTE: Your version of keras does not have the merge object
from keras.layers import Concatenate

from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform

# from keras.engine.topology import Layer
from keras.layers import Layer

from keras import backend as K
K.set_image_data_format('channels_last')
import cv2
from glob import glob
import os
import numpy as np
from numpy import genfromtxt
import pandas as pd
import tensorflow as tf
import tqdm
import cv2
from keras import applications
from keras.models import Model
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Convolution2D, MaxPooling2D,LeakyReLU

# from keras.layers.normalization import BatchNormalization
from keras.layers import BatchNormalization

from keras.metrics import categorical_accuracy
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.preprocessing import image

# from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.applications.resnet import preprocess_input, decode_predictions, ResNet50

from keras.layers import Input, Dense, Flatten, Dropout, Activation, Lambda, Permute, Reshape
import numpy as np
import keras.backend as K
import scipy as sp
from scipy.spatial import distance
from PIL import Image
from keras.preprocessing import image
from sklearn.preprocessing import OneHotEncoder,LabelEncoder

# from hyperas.distributions import choice, uniform, conditional
from hyperas.distributions import choice, uniform # NOTE: had to remove 'conditional'... was removed from the keras main branch...

import hyperopt
from sklearn.model_selection import train_test_split
from keras.utils import Sequence, to_categorical
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import *
from keras.callbacks import (EarlyStopping, LearningRateScheduler,
                             ModelCheckpoint, TensorBoard, ReduceLROnPlateau)
from keras.layers import (Convolution1D, Dense, Dropout, GlobalAveragePooling1D, 
                          GlobalMaxPool1D, Input, MaxPool1D, concatenate)
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input

# from keras.applications.resnet50 import ResNet50  # NOTE: now importing this above
# from keras.applications.resnet50 import preprocess_input  # NOTE: already importing this above

from os import listdir
from pickle import dump
from keras.preprocessing.text import Tokenizer

# from keras.preprocessing.sequence import pad_sequences
from keras.utils import pad_sequences

from keras.layers import Embedding,LSTM,LSTMCell

# from keras.layers.merge import add
from keras.layers import add

from keras.models import Model
from keras.utils import plot_model
from nltk.translate.bleu_score import corpus_bleu #what is a bleu score doing here lol 
from keras.models import load_model
from skimage.io import imread, imshow, imread_collection, concatenate_images
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from __future__ import print_function, division
from builtins import range, input
from keras.preprocessing import image   # for preprocessing the images
from __future__ import print_function, division
from builtins import range,input
from datetime import datetime
import scipy
import keras.utils as image #[C]: Added this here because this is the newest version of tensorflow




def load_preprocess_img(p,shape = None):
    Img = image.load_img(p, target_size=shape)
    # Img = tf.keras.utils.load_img(p, target_size=shape)

    X = image.img_to_array(Img)
    X = np.expand_dims(X,axis=0)    
    X = preprocess_input(X)
    return X

def preprocess_img(frame,shape = None):
    X = np.expand_dims(frame,axis=0)    
    X = preprocess_input(X.astype(('float64')))
    return X
    

#Loading style image
style_img = load_preprocess_img(p='/content/Abstract Style Art.jpeg', shape=(224,224))
batch_shape = style_img.shape
shape = style_img.shape[1:]
shape

shape = (224,224,3)

#customizes keras standard VGG16, get rid of max pooling (because it throws away information) and replace with average pooling layer 
def vgg_avg_pooling(shape):
    vgg = VGG16(input_shape=shape,weights='imagenet',include_top=False)
    model = Sequential()
    for layer in vgg.layers:
        if layer.__class__ == MaxPooling2D:
        # replace it with average pooling    
            model.add(AveragePooling2D())
        else:
            model.add(layer)
    return model   

#[C]: Not really sure what this is doing? 
def vgg_cutoff(shape,num_conv):
    if num_conv<1|num_conv>13:
        print('Error layer must be within range of [1,13]')
    model = vgg_avg_pooling(shape)
    new_model = Sequential()
    n=0
    for layer in model.layers:
        new_model.add(layer)
        if layer.__class__ == Conv2D:
            n+=1
        if n >= num_conv:
            break
    return new_model

#A gram matrix of a set of images represents the similarity (or difference) between two images 
def gram_matrix(img):
    # input is (H, W, C) (C = # feature maps)
    # we first need to convert it to (C, H*W)
    X = K.batch_flatten(K.permute_dimensions(img,(2,0,1)))
    # now, calculate the gram matrix
    # gram = XX^T / N
    gram_mat = K.dot(X,K.transpose(X))/img.get_shape().num_elements()
    return gram_mat 

def style_loss(y,t):
    return K.mean(K.square(gram_matrix(y)-gram_matrix(t)))

#restore original image pixel values generated from VGG-16 
def unpreprocess(img):
    img[..., 0] += 103.939
    img[..., 1] += 116.779
    img[..., 2] += 126.68
    img = img[..., ::-1]
    return img

def scale(x):
    x = x-x.min()
    x=x/x.max()
    return x