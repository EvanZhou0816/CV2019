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
from tqdm import tqdm


class DPCNN():
    def __init__(self, input_tar_path, ranking_matrix,m0, learning_rate=0.01, drop_out=None, hidden_layers=None, epochs=None):
        """
        @params:
            input_tar:a string denotes path of dataset
            ranking_matrix:a matrix in the shape of (5063,300)
        """
        # pre-trained Network(RestNet)
        self.pretrained = keras.applications.ResNet50(weights='imagenet', include_top=False, pooling='avg')
        self.tar_set = tarfile.open(input_tar_path)
        self.rankings = ranking_matrix

        # use of data
        self.m0 = m0

        # Hyper-parameter
        self.learning_rate = learning_rate
        
        self.dropout = drop_out
        self.hidden_layers = hidden_layers
        self.epochs = epochs
        # self.batch_size = batchSz

        # training
        # self.features_matrix = self.forward_pass()
        # self.loss = self.loss_function()
        # self.train = self.optimizer()
        pass

    def forward_pass(self):
        """
            Generate new features matrix (5063,2048)
        """
        tar = self.tar_set
        # self.features_matrix = []
        features_matrix = [[0] * 2048 for i in range(len(tar.getmembers()))]
        i = 0
        # print("=========================Extracting features_matrix=========================")
        # t_bar = tqdm(range(len(tar.getmembers())), total=5063, ascii=True)
        for tar_info in tqdm(tar.getmembers(), ascii=True,desc="Extracting features_matrix"):
            f = tar.extractfile(tar_info)
            f.read()
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
        self.features_matrix = self.forward_pass()
        Loss = 0
        for index in range(len(self.features_matrix)):
            neighbor_indices = random.sample(range(300), 2)
            neighbor_indices.sort()
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

            d_ac = np.linalg.norm(Ia - Ic)
            d_af = np.linalg.norm(Ia - If)
            l = d_ac - d_af + abs(rf - rc) / 300 * self.m0
            Loss += l
        return Loss

    def construct(self):    
        pass

    def optimizer(self):
        # self.features_matrix = self.forward_pass()
        self.loss = self.loss_function()
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
