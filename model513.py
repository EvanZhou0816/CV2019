import tensorflow as tf

import numpy as np
import gzip
import tarfile
import random
import tensorflow.python.keras as keras
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.applications.resnet50 import preprocess_input
from tensorflow.python.keras.applications.vgg19 import VGG19
from tensorflow.python.keras.applications.vgg19 import preprocess_input as vgg_preprocess_input
# from utils.raw_data import process_image

from tqdm import tqdm


class DPCNN():
    def __init__(
        self, input_tar_path, ranking_matrix, index2name, m0=0.1,
            learning_rate=0.01, epochs=None, batchSz=None):
        
        # images placeholder
        self.Ia_place = tf.placeholder(shape=[None, None, None, 3], dtype=tf.float32)
        self.Ic_place = tf.placeholder(shape=[None, None, None, 3], dtype=tf.float32)
        self.If_place = tf.placeholder(shape=[None, None, None, 3], dtype=tf.float32)

        # ranking placeholder
        self.rankings = ranking_matrix
        self.rank_c_f = tf.placeholder(shape=[None], dtype=tf.float32)
        
        self.pretrained_vgg = VGG19(weights='imagenet', pooling='avg', include_top=False)

        # index2name MAP
        self.index2name = index2name
        # imageset
        self.tar_set = tarfile.open(input_tar_path)

        # construct graph
        self.m0 = m0
        self.loss = self.construct_loss()
        self.train = self.optimize()
        print(np.shape(self.rankings)[1])
        pass
    
    def construct(self):
        pass
    
    def construct_loss(self):
        IA = self.pretrained_vgg(self.Ia_place)
        IC = self.pretrained_vgg(self.Ic_place)
        IF = self.pretrained_vgg(self.If_place)

        d_ac = tf.norm(tf.subtract(IA, IC)) #np.linalg.norm(Ia - Ic)
        d_af = tf.norm(tf.subtract(IA, IF)) #np.linalg.norm(Ia - If)
        l = d_ac - d_af + abs(self.rank_c_f[1] - self.rank_c_f[0]) / np.shape(self.rankings)[1] * self.m0
        
        # l = max(l, 0)
        l = tf.nn.relu(l)
        # print("distance(a,c):{} , distance(a,f):{}".format(d_ac, d_af))
        return l
        pass
    
    def optimize(self):
        train = tf.train.AdamOptimizer(learning_rate=0.01).minimize(self.loss)

        # print(train)
        return train
        pass
    
    def generateImage_acf(self, rowa):

        rowc, rowf, rank_c, rank_f = self.generate_nb(rowa)

        Ia_name = self.index2name[rowa]
        obja = self.tar_set.getmember(Ia_name)
        resa = self.tar_set.extractfile(obja)
        # resa = process_image(resa)
        resa = image.load_img(resa, target_size=(224, 224))
        resa = image.img_to_array(resa)
        resa = np.expand_dims(resa, axis=0)
        resa = vgg_preprocess_input(resa)
        Ia_input = resa

        Ic_name = self.index2name[rowc]
        objc = self.tar_set.getmember(Ic_name)
        resc = self.tar_set.extractfile(objc)
        # resc = process_image(resc)
        resc = image.load_img(resc, target_size=(224, 224))
        resc = image.img_to_array(resc)        
        resc = np.expand_dims(resc, axis=0)
        resc = vgg_preprocess_input(resc)
        Ic_input = resc

        If_name = self.index2name[rowf]
        objf = self.tar_set.getmember(If_name)
        resf = self.tar_set.extractfile(objf)
        # resf = process_image(resf)
        resf = image.load_img(resf, target_size=(224, 224))
        resf = image.img_to_array(resf) 
        resf = np.expand_dims(resf, axis=0)
        resf = vgg_preprocess_input(resf)
        If_input = resf

        return Ia_input, Ic_input, If_input, [rank_c, rank_f]
        pass
    
    def generate_nb(self, rowa):
        neighbor_ranks = random.sample(range(np.shape(self.rankings)[1]), 2)
        neighbor_ranks.sort()

        rc = neighbor_ranks[0]
        rf = neighbor_ranks[1]

        index_c = self.rankings[rowa][rc]
        index_f = self.rankings[rowa][rf]

        return index_c, index_f, rc, rf
