import tensorflow as tf
from tensorflow_examples.models.pix2pix import pix2pix


def get_generators():
    impressionism_gen = get_gen("impressionism")
    realism_gen = get_gen("realism")
    romanticism_gen = get_gen("romanticism")
    symbolism_gen = get_gen("symbolism")


    return impressionism_gen, realism_gen, romanticism_gen, symbolism_gen

def get_gen(art):
    impressionism_ckpt_path = "./checkpoints/PROD/impressionism/ckpt-9"
    realism_ckpt_path = "./checkpoints/PROD/realism/ckpt-5"
    romanticism_ckpt_path = "./checkpoints/PROD/romanticism/ckpt-9"
    symbolism_ckpt_path = "./checkpoints/PROD/symbolism/ckpt-9"

    generator_g = pix2pix.unet_generator(3, norm_type='instancenorm')
    generator_f = pix2pix.unet_generator(3, norm_type='instancenorm')

    discriminator_x = pix2pix.discriminator(norm_type='instancenorm', target=False)
    discriminator_y = pix2pix.discriminator(norm_type='instancenorm', target=False)

    generator_g_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    generator_f_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    discriminator_x_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    discriminator_y_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    ckpt = tf.train.Checkpoint(generator_g=generator_g,
                           generator_f=generator_f,
                           discriminator_x=discriminator_x,
                           discriminator_y=discriminator_y,
                           generator_g_optimizer=generator_g_optimizer,
                           generator_f_optimizer=generator_f_optimizer,
                           discriminator_x_optimizer=discriminator_x_optimizer,
                           discriminator_y_optimizer=discriminator_y_optimizer)
    
    if art == "impressionism":
        ckpt.restore(impressionism_ckpt_path).expect_partial()
    if art == "realism":
        ckpt.restore(realism_ckpt_path).expect_partial()
    if art == "romanticism":
        ckpt.restore(romanticism_ckpt_path).expect_partial()
    if art == 'symbolism':
        ckpt.restore(symbolism_ckpt_path).expect_partial()

    return generator_g
        



# normalizing the images to [-1, 1]
def normalize(image):
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1
    return image

def unnormalize(image):
    image = (image + 1) * 127.5
    image = tf.cast(image, tf.uint8)
    return image