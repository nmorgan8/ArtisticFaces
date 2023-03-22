import gradio as gr
import numpy as np 
import tensorflow as tf
import models

def generate_image(image, file, art, choice):
  return image

demo = gr.Interface(
    fn=generate_image,
    inputs=[
        gr.Image(source='webcam', tool=None),
        gr.UploadButton("Click to Upload a File", file_types=["image"]),
        gr.Dropdown(["Rococo", "Romanticism", "Impressionism", "Realism"], label="Art Period"),
        gr.Dropdown(["CycleGANs (UNET)", "CycleGANs (ResNet)", "VGG"], label="Model Choice")
    ],
    outputs=[
        "image"
    ]
)

demo.launch(share=False)