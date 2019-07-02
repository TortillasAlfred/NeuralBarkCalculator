# Neural Bark Calculator
This repository contains a PyTorch based bark calculator for flattened wood log images. Given an input image, the neural network identifies bark and node regions as part of a semantic segmentation task. The input and output images are then combined into a more expressive format :

![Example of network output](res/136.png)

## Installation

First clone the repository with 

``git clone https://github.com/TortillasAlfred/NeuralBarkCalculator.git ``

Then, move inside the project 

``cd NeuralBarkCalculator``

and install all of the project's dependencies with 

``pip3 install -r requirements.txt``

Since PyTorch is not installed in the same way for different OS, we recommend you then install the required PyTorch and torchvision libraries according to the [suggested commands](https://pytorch.org/). For example, on a Windows PC, the command for PyTorch is

``pip3 install https://download.pytorch.org/whl/cu90/torch-1.1.0-cp36-cp36m-win_amd64.whl``

and for torchvision it is

``pip3 install https://download.pytorch.org/whl/cu90/torchvision-0.3.0-cp36-cp36m-win_amd64.whl``

## Usage

The tool lets you predict entire folders at the same time using a single command line instruction. For example, if we want to run the neural network predictions using the computer's CPU, we can type

``python3 src\bark_calculator\predict.py *ROOT_DIR* --device=cpu``

In the above command, ``*ROOT_DIR*`` is considered to be a folder with the following tree structure

```
*ROOT_DIR*
└───samples
│   └───epinette_gelee
│       │   img1.bmp
│       │   img2.bmp
│       │   ...
│   └───epinette_non_gelee
│       │   img3.bmp
│       │   img4.bmp
│       │   ...
...
```

Please note that currently only 3 wood types are supported, namely ``epinette_gelee``, ``epinette_non_gelee`` and ``sapin``. 

The first step of the prediction process is the image preprocessing, where each image is first resized from the expected 4096x4096 format towards a more manageable 1024x1024, before being cut horizontally to trim the usual dark regions above and below the regions of interest. This process is automatically handled by the calculator, which creates a ``processed`` subfolder to the root folder as output for the processed images. 

Once the images are all processed, they are then fed one by one to the neural network, which generates the estimated bark and node regions. The results are all grouped under a ``results`` subfolder, which contains combined images as the one seen above as well as raw outputs. A ``.csv`` file is also created which contains the estimated region percentages for each input image, sorted by name and wood type.

## End notes

This document has been created in order to assist a specific group of users, which are all using Windows. Some commands might be slightly different on a Linux distribution.

Lastly, the code used for training the neural network is all inside the ``__main__.py`` file. 
