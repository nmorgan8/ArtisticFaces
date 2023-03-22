import tensorflow as tf
from tensorflow_examples.models.pix2pix import pix2pix


def get_unet():
    generator_g = pix2pix.unet_generator(3, norm_type='instancenorm')
    generator_f = pix2pix.unet_generator(3, norm_type='instancenorm')

    discriminator_g = pix2pix.discriminator(norm_type='instancenorm', target=False)
    discriminator_f = pix2pix.discriminator(norm_type='instancenorm', target=False)

    generator_g_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    generator_f_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    discriminator_g_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    discriminator_f_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)