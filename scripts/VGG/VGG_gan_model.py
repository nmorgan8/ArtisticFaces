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

tf.compat.v1.disable_eager_execution() #[C]:Placing this code block here to because of keras upgrade

#loading,processing and defining multi_output_model and style loss computation of style image

#Style image load and VGG model load.
path = '/content/Abstract Style Art.jpeg'
img = image.load_img(path)
x = image.img_to_array(img)
x = np.expand_dims(x,axis=0)    
x = preprocess_input(x)

#shape
batch_shape = x.shape
shape = x.shape[1:]
vgg = vgg_avg_pooling(shape)

#Define multi-output model
symb_conv_outputs = [layer.get_output_at(1) for layer in vgg.layers if layer.name.endswith('conv1')]
multi_output_model = Model(vgg.input, symb_conv_outputs)
symb_layer_out = [K.variable(y) for y in multi_output_model.predict(x)]

#Conv layer weight matrix
weights = [0.2,0.4,0.3,0.5,0.2]    
loss=0
#Total style loss
for symb,actual,w in zip(symb_conv_outputs,symb_layer_out,weights):
    loss += w * style_loss(symb[0],actual[0])
    
#gradients which are needed by the optimizer    
grad = K.gradients(loss,multi_output_model.input)
#.function should be lower case 
get_loss_grad = K.function(inputs=[multi_output_model.input], outputs=[loss] + grad)

#Scipy's minimizer function(fmin_l_bfgs_b) allows us to pass back function value f(x) and 
#its gradient f'(x), which we calculated in earlier step. 
#However, we need to unroll the input to minimizer function in1-D array format and both loss and gradient must be np.float64.

def get_loss_grad_wrapper(x_vec):
    l,g = get_loss_grad([x_vec.reshape(*batch_shape)])
    return l.astype(np.float64), g.flatten().astype(np.float64) 

#Function to minimize loss
def min_loss(fn,epochs,batch_shape):
    t0 = datetime.now()
    losses = []
    x = np.random.randn(np.prod(batch_shape))
    for i in range(epochs):
        x, l, _ = scipy.optimize.fmin_l_bfgs_b(func=fn,x0=x,maxfun=20)
    # bounds=[[-127, 127]]*len(x.flatten())
    #x = np.clip(x, -127, 127)
    # print("min:", x.min(), "max:", x.max())
        print("iter=%s, loss=%s" % (i, l))
        losses.append(l)
    print("duration:", datetime.now() - t0)
    plt.plot(losses)
    plt.show()

    newimg = x.reshape(*batch_shape)
    final_img = unpreprocess(newimg)
    return final_img[0]   

#Check if Output directory exists, if not, create a new output directory
import os

DIR = '/content/style_images_output'
if not os.path.isdir(DIR):
  os.makedirs(DIR)
  print("created folder: ", DIR)
else: 
  print(DIR, "folder already exists.")