import tensorflow as tf
from tensorflow_examples.models.pix2pix import pix2pix
import matplotlib.pyplot as plt


def get_model():
    generator_g = pix2pix.unet_generator(3, norm_type='instancenorm')
    generator_f = pix2pix.unet_generator(3, norm_type='instancenorm')

    discriminator_x = pix2pix.discriminator(norm_type='instancenorm', target=False)
    discriminator_y = pix2pix.discriminator(norm_type='instancenorm', target=False)

    generator_g_optimizer = tf.keras.optimizers.legacy.Adam(2e-4, beta_1=0.5)
    generator_f_optimizer = tf.keras.optimizers.legacy.Adam(2e-4, beta_1=0.5)

    discriminator_x_optimizer = tf.keras.optimizers.legacy.Adam(2e-4, beta_1=0.5)
    discriminator_y_optimizer = tf.keras.optimizers.legacy.Adam(2e-4, beta_1=0.5)

    ckpt = tf.train.Checkpoint(generator_g=generator_g,
                           generator_f=generator_f,
                           discriminator_x=discriminator_x,
                           discriminator_y=discriminator_y,
                           generator_g_optimizer=generator_g_optimizer,
                           generator_f_optimizer=generator_f_optimizer,
                           discriminator_x_optimizer=discriminator_x_optimizer,
                           discriminator_y_optimizer=discriminator_y_optimizer)

    checkpoint_path = "./rococo_checkpoints/train/ckpt-12"

    ckpt.restore(checkpoint_path).expect_partial()

    return generator_g, generator_f, discriminator_x, discriminator_y, generator_g_optimizer, generator_f_optimizer, discriminator_x_optimizer, discriminator_y_optimizer

def get_vgg(style_img):
    pass


# normalizing the images to [-1, 1]
def normalize(image):
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1
    return image

def unnormalize(image):
    image = (image + 1) * 127.5
    image = tf.cast(image, tf.uint8)
    return image
