import gradio as gr
import numpy as np 
import tensorflow as tf
from scripts.CycleGAN import cycle_gan_model
from scripts.VGG.test import get_model, tensor_to_image, load_input, get_style

generator_g, _, _, _, _, _, _, _ = cycle_gan_model.get_model()
vgg_model = get_model()


def generate_image(inp, file, art, choice):
    if choice == "VGG":
        inp = load_input(inp)
        style_img = get_style(art)
        stylized_image = vgg_model(tf.constant(inp), tf.constant(style_img))[0]
        return tensor_to_image(stylized_image)

    elif choice == "CycleGANs (UNET)":
        inp = cycle_gan_model.normalize(inp)
        inp = tf.expand_dims(inp, axis=0)
        image = generator_g(inp)
        image = cycle_gan_model.unnormalize(image)
        return image.numpy()[0]

demo = gr.Interface(
    fn=generate_image,
    inputs=[
        gr.Image(source='webcam', tool=None, shape=None),
        gr.UploadButton("Click to Upload a File", file_types=["image"]),
        gr.Dropdown(["Rococo", "Romanticism", "Impressionism", "Realism"], label="Art Period"),
        gr.Dropdown(["CycleGANs (UNET)", "CycleGANs (ResNet)", "VGG"], label="Model Choice")
    ],
    outputs=[
        "image"
    ]
)

demo.launch(share=False)

