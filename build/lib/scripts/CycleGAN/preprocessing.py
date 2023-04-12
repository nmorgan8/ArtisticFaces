import tensorflow as tf

# normalizing the images to [-1, 1]
def normalize(image):
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1
    return image

def unnormalize(image):
    image = (image + 1) * 127.5
    image = tf.cast(image, tf.uint8)
    return image
