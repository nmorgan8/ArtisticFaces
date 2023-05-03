import gradio as gr
import numpy as np 
import tensorflow as tf
import PIL
from scripts.CycleGAN import cycle_gan_model
from scripts.VGG.test import get_model, tensor_to_image, load_input, get_style

impressionism_gen, realism_gen, romanticism_gen, symbolism_gen = cycle_gan_model.get_generators()
vgg_model = get_model()

examples = [
    ['data/impressionism_vgg_sample.jpeg', 'Impressionism'],
    ['data/symbolism_vgg_sample.jpeg', 'Symbolism'],
    ['data/romanticism_vgg_sample.jpeg', 'Romanticism'],
    ['data/realism_vgg_sample.jpeg', 'Realism'],
]


def generate_image(inp, art, choice):
    if choice == "VGG":
        inp = load_input(inp)
        style_img = get_style(art)
        stylized_image = vgg_model(tf.constant(inp), tf.constant(style_img))[0]
        return tensor_to_image(stylized_image)

    elif choice == "CycleGANs (UNET)":
        inp = tf.image.resize(inp, [256, 256])
        inp = tf.expand_dims(inp, axis=0)
        inp = cycle_gan_model.normalize(inp)
        if art == "Impressionism":
            image = impressionism_gen(inp)
        if art == "Symbolism":
            image = symbolism_gen(inp)
        if art == "Romanticism":
            image = romanticism_gen(inp)
        if art == "Realism":
            image = realism_gen(inp)
        image = cycle_gan_model.unnormalize(image)
        return image.numpy()[0]
    

webcam = gr.Interface(
    fn=generate_image,
    inputs=[
        gr.Image(source='webcam', label='Style Image'),
        gr.Dropdown(["Impressionism", "Symbolism", "Romanticism", "Realism"], label="Art Period"),
        gr.Dropdown(["CycleGANs (UNET)", "VGG"], label="Model Choice")
    ],
    outputs=[
        "image"
    ],
    title="Artistic Faces",
    description='<p style="text-align: center;">Transform your face and surroundings to your favorite art movement!</p>',
    examples=examples
)

upload = gr.Interface(
    fn=generate_image,
    inputs=[
        gr.Image(source='upload', label='Style Image'),
        gr.Dropdown([ "Impressionism", "Symbolism", "Romanticism", "Realism"], label="Art Period"),
        gr.Dropdown(["CycleGANs (UNET)", "VGG"], label="Model Choice")
    ],
    outputs=[
        "image"
    ],
    title="Artistic Faces",
    description='<p style="text-align: center;">Transform your face and surroundings to your favorite art movement!</p>',
    examples=examples
)

demo = gr.TabbedInterface([webcam, upload], ["Webcam", "Upload"])

demo.launch(share=True)

