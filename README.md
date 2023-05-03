# ArtisticFaces

## Overview
This repo contains code for a machine learning based platform for transfering historical art styles like impressionism and realism into a user's uploaded/captured images in real-time using state-of-the-art using two different appraaches: 
 - Training and fine-tuning a Cyclical Generative Adversarial Network (CycleGAN) on ImageNet and Wikiart dataset
 - Utilizing layers in VGG - 16 to capture style information for style transference
 
## Structure: 
 - src: contains run.py file to launch gradio interface and interact with our application
 - scripts: contains CycleGAN and VGG-16 related code and dependencies 
  
## Running our Gradio interface
1. git clone this repo
````
git clone 
```

2. cd into directory 
```
cd Artistic Faces
```

3. install project requirements and set module dependencies 
```
pip install -r requirements.txt
python setup.py install 
```

4. launch gradio app interface to interact with our models 
```
python run.py 
```


## Team Members
 - Nick Morgan
 - Claudia Gusti

## Technologies
 - CycleGANs, VGG-Net 
 - Tensorflow, Keras, matplotlib, Gradio

## Mini-Goals
 - Develop POC with one art movement
 - Test different art movements over longer epochs of training
 - Build basic pipeline in Gradio
