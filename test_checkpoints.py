import models
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import tensorflow as tf

def generate_images(model, test_input):
  print(test_input.shape)
  print(test_input)
  prediction = model(test_input)
  print(prediction.shape)
  print(prediction)
    
  plt.figure(figsize=(12, 12))

  display_list = [test_input[0], prediction[0]]
  title = ['Input Image', 'Predicted Image']

  for i in range(2):
    plt.subplot(1, 2, i+1)
    plt.title(title[i])
    # getting the pixel values between [0, 1] to plot it.
    plt.imshow(display_list[i] * 0.5 + 0.5)
    plt.axis('off')
  plt.show(block=True)

generator_g, generator_f, discriminator_x, discriminator_y, generator_g_optimizer, generator_f_optimizer, discriminator_x_optimizer, discriminator_y_optimizer = models.get_model()

test_img = Image.open('./image.png')
test_img = tf.convert_to_tensor(test_img)
test_img = models.normalize(test_img)
test_img = tf.reshape(test_img, [1, 256, 256, 3])

generate_images(generator_g, test_img)