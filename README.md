# ArtisticFaces

## Overview

This repo contains code for a machine learning based platform for transfering historical art styles like impressionism and realism into a user's uploaded/captured images in real-time using two different appraaches:

- Training and fine-tuning a Cyclical Generative Adversarial Network (CycleGAN) on ImageNet and Wikiart dataset
- Utilizing layers in VGG - 16 to capture style loss and information for style transference

## Project Structure:

- src: contains run.py file to launch gradio interface and interact with our application
- scripts: contains CycleGAN and VGG-16 related code and dependencies
- notebooks: contains .ipnyb notebooks for debugging and demo purposes
- training: training script for CycleGAN
- data: contains compressed files to image data used to train

## Getting Started

To get a local copy up and running, please follow these simple steps.

### Prerequisites

Here is what you need to run ArtisiticFaces.

- Python3
- pip
- Conda (use GPU on Windows machine)

### Setup Repository

1. git clone this repo

```
git clone https://github.com/nmorgan8/ArtisticFaces.git
```

2. cd into directory

```
cd ArtisticFaces
```

### Setup Python Modules

1. Install project requirements and set module dependencies

```
pip install -r requirements.txt
```

2. Install tensorflow_examples

```
git clone https://github.com/tensorflow/examples.git
```

3. Set module dependencies

```
python3 setup.py install
```

### Untar Model Checkpoints

1. Untar PROD.tar.gz to retrieve trained models in checkpoints directory

```

```

### Launch Gradio

```
python3 src/run.py
```

### Common Errors

1. ValueError: Trying to load a model of incompatible/unknown type. <file_name>

```
rm -r <file_name>
python3 src/run.py
```

## Team Members

- Nick Morgan
- Claudia Gusti

## Technologies

- CycleGANs, VGG-Net
- Tensorflow, Keras, matplotlib, Gradio
