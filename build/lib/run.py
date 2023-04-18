import run as gr
import numpy as np 
import tensorflow as tf
import scripts
from scripts.CycleGAN import cycle_gan_model

generator_g, _, _, _, _, _, _, _ = cycle_gan_model.get_model()

def generate_image(inp, file, art, choice):
    inp = cycle_gan_model.normalize(inp)
    inp = tf.expand_dims(inp, axis=0)
    image = generator_g(inp)
    image = cycle_gan_model.unnormalize(image)
    return image.numpy()[0]

demo = gr.Interface(
    fn=generate_image,
    inputs=[
        gr.Image(source='webcam', tool=None, shape=(256, 256)),
        gr.UploadButton("Click to Upload a File", file_types=["image"]),
        gr.Dropdown(["Rococo", "Romanticism", "Impressionism", "Realism"], label="Art Period"),
        gr.Dropdown(["CycleGANs (UNET)", "CycleGANs (ResNet)", "VGG"], label="Model Choice")
    ],
    outputs=[
        "image"
    ]
)

demo.launch(share=False)