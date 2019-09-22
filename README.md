# Classifying Flowers With Transfer Learning: An AI application for classifying British categories of flowers

This project is part of the Data Scientist Nanodegree Program and consists of training an image classifier to recognize different species of flowers. Moreover, the classifier will be converted in an application fashion, which means one could run the code in the terminal to make predictions with the training model for a given image.

## Motivation
The motivation of this project is to used the concepts of Deep Learnd and Transfer Learning, creating an application with real data.  Moreover, this project aims to allow one to be familiar with the PyTorch open source library for computer vision and NLP.

## Documentation
Please, find a post on this project on [Medium]().

## The Dataset
The data being used in this project was extracted from the Visual Geometric Group from the University of Oxford and can be accessed in here.
The data contains a 102 category dataset regarding flower categories commonly occurring in the United Kingdom. As explained in the original source, each flower category consists of between 40 and 258 images.

## Files in this repository
- cat_to_name.json: a file mapping each category to its correspondig flower name
- checkpoint.pth: the trained model
- classifiers.py: python file with the Neural Network classifier
- data_treatment.py: python module for treating the images
- predict.py: python module for making predictions
- train.py: python module for training the network
- workspace-utils.py: utils function for the jupyter notebook
- Image Classifier Project.ipynb: a Jupyter Notebook with preparation code
- Image Classifier Project.html: an html version of the Jupyter Notebook

## How to Run This Code:

- **Training Data**
In order to train the model, one should give the input path for the images as well as optional arguments, such as the pre-trained model to be used (vgg11, vgg13 or vgg16), whether or not to use a GPU, the desired learning rate and also the size of the hidden layers.

 - python train.py './flowers'

 - python train.py './flowers' --save_dir './'

 - python train.py './flowers' --arch 'vgg11'

 - python train.py './flowers' -l 0.005 -e 2 -a 'vgg13'

 - python train.py './flowers' -l 0.005 -e 2 -a 'vgg16' --gpu

 - python train.py './flowers' --learning_rate 0.01 --hidden_units 2048 1024 --epochs 20 --gpu

- **Predicting Flowers**
In order to make predictions, one should give the input path for the image to be predicted and the path to the trained model (the checkpoint). Moreover, a .json file with the corresponding name of the flowers for each class (category) can be passed as an argument.

 - python predict.py './flowers/test/15/image_06351.jpg' 'checkpoint.pth'

 - python predict.py './flowers/test/15/image_06351.jpg' 'checkpoint.pth' --category_names cat_to_name.json

 - python predict.py './flowers/test/15/image_06351.jpg' 'checkpoint.pth' --category_names cat_to_name.json --top_k 3

 - python predict.py './flowers/test/15/image_06351.jpg' 'checkpoint.pth' --category_names cat_to_name.json --top_k 5 --gpu

## Used Libraries
- json
- os
- numpy
- matplotlib.pyplot
- torch
- torch.nn
- torch.optim
- torch.nn.functional
- torchvision.datasets
- torchvision.transforms
- torchvision.models
- PIL.Image

## Acknowledgements
I would like to ackonwledge Udacity for providing materials as part of the Data Scientist Nanodegree and the Visual Geometric Group from the University of Oxford for providing the dataset for the public.
