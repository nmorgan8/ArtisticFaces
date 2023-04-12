import keras.utils as image
import numpy as np
from keras.applications.resnet import preprocess_input, decode_predictions, ResNet50



def load_preprocess_img(p,shape = None):
    Img = image.load_img(p, target_size=shape)
    X = image.img_to_array(Img)
    X = np.expand_dims(X,axis=0)    
    X = preprocess_input(X)
    return X

#restore original image pixel values generated from VGG-16 
def unpreprocess(img):
    img[..., 0] += 103.939
    img[..., 1] += 116.779
    img[..., 2] += 126.68
    img = img[..., ::-1]
    return img

#testing load style image
style_img = load_preprocess_img(p='/Users/claudiagusti/Desktop/Spring2022/CMSI 6998/ArtisticFaces/scripts/VGG/Abstract Style Art.jpeg', shape=(224,224))
batch_shape = style_img.shape
shape = style_img.shape[1:]
print(shape)

