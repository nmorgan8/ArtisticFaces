from keras.applications.vgg16 import VGG16
from keras.models import Sequential
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras import backend as K

def vgg_avg_pooling(shape):
    """customizes keras standard VGG16, get rid of max pooling (because it throws away information) and replace with average pooling layer """
    vgg = VGG16(input_shape=shape,weights='imagenet',include_top=False)
    model = Sequential()
    for layer in vgg.layers:
        if layer.__class__ == MaxPooling2D:
        # replace it with average pooling    
            model.add(AveragePooling2D())
        else:
            model.add(layer)
    return model   

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

def gram_matrix(img):
    """A gram matrix of a set of images represents the similarity (or difference) between two images"""
    # input is (H, W, C) (C = # feature maps)
    # we first need to convert it to (C, H*W)
    X = K.batch_flatten(K.permute_dimensions(img,(2,0,1)))
    # now, calculate the gram matrix
    # gram = XX^T / N
    gram_mat = K.dot(X,K.transpose(X))/img.get_shape().num_elements()
    return gram_mat 

def style_loss(y,t):
    return K.mean(K.square(gram_matrix(y)-gram_matrix(t)))

  