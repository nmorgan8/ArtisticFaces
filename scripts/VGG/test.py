"""
This script uses tensorflow hub to download ta neural net transfer model and runs inference on a content and style image to produce an image output
Author: Claudia G
Date: 4/16/23
"""



import os
import tensorflow as tf
# Load compressed models from tensorflow_hub
os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'
# import IPython.display as display

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (12, 12)
mpl.rcParams['axes.grid'] = False

import numpy as np
import PIL.Image
import time
import functools
import tensorflow_hub as hub 


def tensor_to_image(tensor):
  tensor = tensor*255
  tensor = np.array(tensor, dtype=np.uint8)
  if np.ndim(tensor)>3:
    assert tensor.shape[0] == 1
    tensor = tensor[0]
  return PIL.Image.fromarray(tensor)

content_path = tf.keras.utils.get_file('YellowLabradorLooking_new.jpg', 'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg')
style_path = tf.keras.utils.get_file('kandinsky5.jpg','https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg')

def load_img(path_to_img):
  max_dim = 512
  img = tf.io.read_file(path_to_img)
  img = tf.image.decode_image(img, channels=3)
  img = tf.image.convert_image_dtype(img, tf.float32)

  shape = tf.cast(tf.shape(img)[:-1], tf.float32)
  long_dim = max(shape)
  scale = max_dim / long_dim

  new_shape = tf.cast(shape * scale, tf.int32)

  img = tf.image.resize(img, new_shape)
  img = img[tf.newaxis, :]
  return img

def load_input(img):
  max_dim = 512
  img = tf.image.convert_image_dtype(img, tf.float32)
  shape = tf.cast(tf.shape(img)[:-1], tf.float32)
  long_dim = max(shape)
  scale = max_dim / long_dim

  new_shape = tf.cast(shape * scale, tf.int32)

  img = tf.image.resize(img, new_shape)
  img = img[tf.newaxis, :]
  return img

def imshow(image, title=None):
  if len(image.shape) > 3:
    image = tf.squeeze(image, axis=0)

  plt.imshow(image)
  if title:
    plt.title(title)

def get_model():
  hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
  return hub_model

def get_style(style):
  if style == "Symbolism":
    return load_img('data/symbolism_vgg_sample.jpeg')
  elif style == "Romanticism":
    return load_img('data/romanticism_vgg_sample.jpeg')
  elif style == "Impressionism":
    return load_img('data/impressionism_vgg_sample.jpeg')
  elif style == "Realism":
    return load_img('data/realism_vgg_sample.jpeg')

# content_image = load_img(content_path)
# style_image = load_img(style_path)

# plt.subplot(1, 2, 1)
# print('showing content image')
# imshow(content_image, 'Content Image')

# plt.subplot(1, 2, 2)
# print("showing style image")
# imshow(style_image, 'Style Image')

#Inference
# hub_model = get_model()
# stylized_image = hub_model(tf.constant(content_image), tf.constant(style_image))[0]
# print('printing final image..')
# plt.imshow(tensor_to_image(stylized_image))
# plt.show()

vgg_model = get_model()