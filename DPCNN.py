import tensorflow as tf
import keras
import numpy as np
import gzip
import tarfile
import tensorflow.python.keras as keras
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense, Dropout, Flatten
from tensorflow.python.keras.applications.resnet50 import preprocess_input 
from scipy.spatial import distance
from sklearn import metrics
from utils import diffusion



class DPCNN():
    def __init__(self, inputs, output_size, learning_rate, drop_out, hidden_layers, epochs, batchSz, m0, t, k):

        # pre-trained Network(RestNet)
        self.pretrained = keras.applications.ResNet50(weights='imagenet', include_top=False, pooling='avg')
        self.inputs = inputs

        # use of data
        self.m0 = m0

        # Hyper-parameter
        self.learning_rate = learning_rate
        self.output_size = output_size
        self.dropout = drop_out
        self.hidden_layers = hidden_layers
        self.epochs = epochs
        self.batch_size = batchSz
        
        #training
        self.loss = self.loss_function()

        # # computation graph construction
        self.triplelets   #
        pass

    def forward_pass(self):
        #parameters unsettled

        return probabilities

    def loss_function(self):

        L = d1-d2+abs(r1-r2)/300*self.m0

        return L

    def construct(self):
        pass

    def optimizer(self):
        train = tf.train.AdamOptimizer(learning_rate=0.01).minimize(self.loss)
        return train

    def train(self, verbose=0):
        
        pass

    def predict(self, inputs, verbose=0):
        pass

    def test(self, test_inputs, test_labels, verbose=0):
        pass
