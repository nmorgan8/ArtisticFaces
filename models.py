import tensorflow as tf
from tensorflow_examples.models.pix2pix import pix2pix
import os


def get_unet():
    generator_g = pix2pix.unet_generator(3, norm_type='instancenorm')
    generator_f = pix2pix.unet_generator(3, norm_type='instancenorm')

    discriminator_x = pix2pix.discriminator(norm_type='instancenorm', target=False)
    discriminator_y = pix2pix.discriminator(norm_type='instancenorm', target=False)

    generator_g_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    generator_f_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    discriminator_x_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    discriminator_y_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    return generator_g, generator_f, discriminator_x, discriminator_y, generator_g_optimizer, generator_f_optimizer, discriminator_x_optimizer, discriminator_y_optimizer

generator_g, generator_f, discriminator_x, discriminator_y, generator_g_optimizer, generator_f_optimizer, discriminator_x_optimizer, discriminator_y_optimizer = get_unet()

ckpt = tf.train.Checkpoint(generator_g=generator_g,
                           generator_f=generator_f,
                           discriminator_x=discriminator_x,
                           discriminator_y=discriminator_y,
                           generator_g_optimizer=generator_g_optimizer,
                           generator_f_optimizer=generator_f_optimizer,
                           discriminator_x_optimizer=discriminator_x_optimizer,
                           discriminator_y_optimizer=discriminator_y_optimizer)

checkpoint_path = "./rococo_checkpoints/ckpt-11.data-00000-of-00001"

ckpt.read(checkpoint_path)
