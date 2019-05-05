import tensorflow as tf
import keras
import numpy as np
import gzip
import tarfile
import random
import tensorflow.python.keras as keras
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense, Dropout, Flatten
from tensorflow.python.keras.applications.resnet50 import preprocess_input
from utils.raw_data import process_image
from scipy.spatial import distance
from sklearn import metrics


class DPCNN():
    def __init__(self, input_tar, ranking_matrix, learning_rate = 0.01, drop_out=None, hidden_layers = None, epochs, m0):
        """
        @params:
            input_tar:a string denotes path of dataset
            ranking_matrix:a matrix in the shape of (5063,300)
        """
        # pre-trained Network(RestNet)
        self.pretrained = keras.applications.ResNet50(weights='imagenet', include_top = False, pooling='avg')
        self.tar_set = tarfile.open(input_tar)
        self.rankings = ranking_matrix

        # use of data
        self.m0 = m0

        # Hyper-parameter
        self.learning_rate = learning_rate
        self.output_size = output_size
        self.dropout = drop_out
        self.hidden_layers = hidden_layers
        self.epochs = epochs
        # self.batch_size = batchSz   

        # training
        self.loss = self.loss_function()
        self.features_matrix = self.forward_pass()
        self.train = self.optimizer()
        pass

    def forward_pass(self):
        """
            Generate new features matrix (5063,2048)
        """
        tar = self.tar_set
        # self.features_matrix = []
        features_matrix = [[0]*2048 for i in range(5063)]
        i = 0
        for tar_info in tar.getmembers():
            f = tar.extractfile(tar_info)
            res = process_image(f)
            res = np.expand_dims(res, axis=0)
            res = preprocess_input(res)
            features = self.pretrained.predict(res)
            # features_reduce = features.squeeze()
            # print("feature shape: ",np.shape(features))
            features_matrix[i] = features
            i += 1
        return features_matrix

    def loss_function(self):
        """
        @descriptions:
            #1 Get features of Ia,Ic,If
            #2 Calculate d(a,c),d(a,f);Get rf,rc
            #3 Calculate Triplet loss function by:
            L = sigma{ d(Ia,Ic) - d(Ia,If) + abs(rf-rc)/300 * m0 } 
        """
        Loss = 0
        for index in range(len(self.features_matrix)):
            neighbor_indices = random.sample(range(300), 2)
            indices.sort() 
            rc = neighbor_indices[0]
            rf = neighbor_indices[1]

            # Get Ia's feature
            Ia = self.features_matrix[index]
            Ia = np.array(Ia)
            # Get Ic's feature
            index_c = self.rankings[index][rc]
            Ic = self.features_matrix[index_c]
            Ic = np.array(Ic)
            # Get If's feature
            index_f = self.rankings[index][rf]
            If = self.features_matrix[index_f]
            If = np.array(If)

            d_ac = np.linalg.norm(Ia-Ic)
            d_af = np.linalg.norm(Ia-If)
            l = d_ac - d_af + abs(rf-rc)/300 * self.m0
            Loss += l
        return Loss

    def construct(self):    
        pass

    def optimizer(self):
        train = tf.train.AdamOptimizer(learning_rate=0.01).minimize(self.loss)
        return train

    def train(self, verbose=0):
        
        pass

    def predict(self, input_image, verbose=0):
        res = process_image(input_image)
        res = np.expand_dims(res, axis=0)
        res = preprocess_input(res)
        features = self.pretrained.predict(res)
        return features
        pass

    def test(self, test_inputs, test_labels, verbose=0):
        pass
