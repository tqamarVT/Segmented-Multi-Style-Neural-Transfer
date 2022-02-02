! pip install git+https://github.com/cocodataset/panopticapi.git
import os
import requests
import io
import math
from google.colab import files
import time
import tensorflow as tf
import keras as keras
from tensorflow.keras.applications import VGG19
from keras.applications.vgg19 import preprocess_input
from keras import Model
from tensorflow.keras.optimizers import Adam
import IPython.display as display
import PIL
from PIL import Image
import keras.backend as K
import numpy as np
from tensorflow.keras import layers as Layers
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from matplotlib import pyplot as plt
import torch
from torch import nn
from torchvision.models import resnet50
import torchvision.transforms as T
torch.set_grad_enabled(False);
import panopticapi
from panopticapi.utils import id2rgb, rgb2id
import itertools
from itertools import chain
import seaborn as sns
import cv2
