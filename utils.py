'''
=========================

Common utility functions(Parsing/formatting etc.)

=========================
'''
import json
import keras
import numpy as np 

from keras.preprocessing import image
from keras import backend as kb
import tqdm

'''
Shuffle indices vector
'''
def shuffle_indices(index_range):
    if isinstance(index_range, (list, np.ndarray)):
        vec = index_range
    else:
        vec = np.arange(index_range)
    
    np.random.shuffle(vec)
    return vec

'''
Data generator for keras models
This will allow us to load in images and generate batch data on the fly, solving the problem
of running out of memory when trying to load in too many images
(order of input data MUST be [meta, image classification, image path]!!!)
'''
class DataGen(keras.utils.Sequence):
    def __init__(self,inputs):
        self.inputs = inputs
        pass
    def __data_generation(self):
        outputs=[]
        img_array=[]
        img = image.load_img(path="")
        img_array = image.img_to_array(img)
        return outputs