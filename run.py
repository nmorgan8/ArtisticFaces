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

descriptions = {
    "Impressionism": "Impressionism developed in France in the nineteenth century and is based on the practice of painting out of doors and spontaneously 'on the spot' rather than in a studio from sketches. Main impressionist subjects were landscapes and scenes of everyday life.",
    "Symbolism": "Symbolism was both an artistic and a literary movement that suggested ideas through symbols and emphasized the meaning behind the forms, lines, shapes, and colors. The works of some of its proponents exemplify the ending of the tradition of representational art coming from Classical times. Symbolism can also be seen as being at the forefront of modernism, in that it developed new and often abstract means to express psychological truth and the idea that behind the physical world lay a spiritual reality.",
    "Romanticism": "Romantic art focused on emotions, feelings, and moods of all kinds including spirituality, imagination, mystery, and fervor. The subject matter varied widely including landscapes, religion, revolution, and peaceful beauty. The brushwork for romantic art became looser and less precise.",
    "Realism": "In its specific sense realism refers to a mid nineteenth century artistic movement characterised by subjects painted from everyday life in a naturalistic manner; however the term is also generally used to describe artworks painted in a realistic almost photographic way.",
}


def generate_image(inp, art, choice):
    if choice == "VGG":
        inp = load_input(inp)
        style_img = get_style(art)
        stylized_image = vgg_model(tf.constant(inp), tf.constant(style_img))[0]
        desc = descriptions[art]
        return tensor_to_image(stylized_image), desc

    elif choice == "CycleGANs (UNET)":
        inp = tf.image.resize(inp, [256, 256])
        inp = tf.expand_dims(inp, axis=0)
        inp = cycle_gan_model.normalize(inp)
        desc = descriptions[art]
        if art == "Impressionism":
            image = impressionism_gen(inp)
        if art == "Symbolism":
            image = symbolism_gen(inp)
        if art == "Romanticism":
            image = romanticism_gen(inp)
        if art == "Realism":
            image = realism_gen(inp)
        image = cycle_gan_model.unnormalize(image)
        return image.numpy()[0], desc
    

webcam = gr.Interface(
    fn=generate_image,
    inputs=[
        gr.Image(sources=['webcam'], label='Style Image'),
        gr.Dropdown(["Impressionism", "Symbolism", "Romanticism", "Realism"], label="Art Period"),
        gr.Dropdown(["VGG", "CycleGANs (UNET)"], label="Model Choice")
    ],
    outputs=[
        gr.Image(label='Stylized Image'),
        gr.Textbox(label='Art Style Information')
    ],
    title="Artistic Faces",
    description='<p style="text-align: center;">Transform your face and surroundings to your favorite art movement!</p>',
    examples=examples
)

upload = gr.Interface(
    fn=generate_image,
    inputs=[
        gr.Image(sources=['upload'], label='Style Image'),
        gr.Dropdown([ "Impressionism", "Symbolism", "Romanticism", "Realism"], label="Art Period"),
        gr.Dropdown(["VGG", "CycleGANs (UNET)"], label="Model Choice")
    ],
    outputs=[
        gr.Image(label='Stylized Image'),
        gr.Textbox(label='Art Style Information')
    ],
    title="Artistic Faces",
    description='<p style="text-align: center;">Transform your face and surroundings to your favorite art movement!</p>',
    examples=examples
)

demo = gr.TabbedInterface([webcam, upload], ["Webcam", "Upload"])

demo.launch()

