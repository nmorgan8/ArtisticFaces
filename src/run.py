import gradio as gr
import numpy as np 
import tensorflow as tf
from scripts.CycleGAN import cycle_gan_model
from scripts.VGG.test import get_model, tensor_to_image, load_input, get_style

generator_g, _, _, _, _, _, _, _ = cycle_gan_model.get_model()
vgg_model = get_model()

examples = [
    ['data/impressionism_vgg_sample.jpeg', 'Impressionism'],
    ['data/rococo_vgg_sample.jpeg', 'Rococo'],
    ['data/romanticism_vgg_sample.jpeg', 'Romanticism'],
    ['data/realism_vgg_sample.jpeg', 'Realism'],
]


def generate_webcam(inp, art, choice):
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
    
def generate_file(file, art, choice):
    if choice == "VGG":
        file = load_input(file)
        style_img = get_style(art)
        stylized_image = vgg_model(tf.constant(file), tf.constant(style_img))[0]
        return tensor_to_image(stylized_image)

    elif choice == "CycleGANs (UNET)":
        file = cycle_gan_model.normalize(file)
        file = tf.expand_dims(file, axis=0)
        image = generator_g(file)
        image = cycle_gan_model.unnormalize(image)
        return image.numpy()[0]

webcam = gr.Interface(
    fn=generate_webcam,
    inputs=[
        gr.Image(source='webcam', label='Style Image'),
        gr.Dropdown([ "Impressionism", "Rococo", "Romanticism", "Realism"], label="Art Period"),
        gr.Dropdown(["VGG", "CycleGANs (UNET)"], label="Model Choice")
    ],
    outputs=[
        "image"
    ],
    title="Artistic Faces",
    description='<p style="text-align: center;">Transform your face and surroundings to your favorite art movement!</p>',
    examples=examples
)

upload = gr.Interface(
    fn=generate_file,
    inputs=[
        gr.Image(source='upload', label='Style Image'),
        gr.Dropdown([ "Impressionism", "Rococo", "Romanticism", "Realism"], label="Art Period"),
        gr.Dropdown(["VGG", "CycleGANs (UNET)"], label="Model Choice")
    ],
    outputs=[
        "image"
    ],
    title="Artistic Faces",
    description='<p style="text-align: center;">Transform your face and surroundings to your favorite art movement!</p>',
    examples=examples
)

demo = gr.TabbedInterface([webcam, upload], ["Webcam", "Upload"])

demo.launch(share=False)

