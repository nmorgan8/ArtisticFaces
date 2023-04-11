import gradio as gr
import numpy as np 
import tensorflow as tf
import models

generator_g, generator_f, discriminator_x, discriminator_y, generator_g_optimizer, generator_f_optimizer, discriminator_x_optimizer, discriminator_y_optimizer = models.get_model()

def generate_image(inp, file, art, choice):
    inp = inp.reshape((-1, 256, 256, 3))
    inp = models.normalize(inp)
    image = generator_g(inp)
    image = models.unnormalize(image)
    return image.numpy()[0]

def display_img(inp, file, art, choice):
    return inp

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