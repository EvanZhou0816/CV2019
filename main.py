import numpy as np
import gzip
import tarfile
import os
from utils.raw_data import process_image
from PIL import Image
from tqdm import tqdm
import tensorflow as tf
import tensorflow.python.keras as keras
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense, Dropout, Flatten
from tensorflow.python.keras.applications.resnet50 import preprocess_input 
from DPCNN import DPCNN


if __name__ == "__main__":
    print(os.getcwd())
    os.chdir('F:/workspace_Python/CV2019')
    tarpath = "../oxbuild_images.tgz"
    tar = tarfile.open("../oxbuild_images.tgz")
    # ResNet_model = keras.applications.ResNet50(weights='imagenet', include_top=False, pooling='avg')
    # outputs1 = ResNet_model.get_layer('activation_48').output
    # outputs1 = Dense(4096, activation='softmax')(outputs1)
    # extracter = Model(inputs=ResNet_model.input,outputs=outputs1)  
    # for layer in ResNet_model.layers:
    #     print(layer.name, layer.trainable)
    
    rankings = np.load("npy/ranking5.npy")
    CNN_model = DPCNN(tarpath, rankings, 10)
    
    sess = tf.Session()
    saver = tf.train.Saver(max_to_keep=2)
    init = tf.global_variables_initializer()
    writer = tf.summary.FileWriter("logs/", sess.graph)
    sess.run(init)
 
    for i in range(10):
        print("\n========================= epoch being processed:{} =========================".format(i))
        l, _ = sess.run(CNN_model.loss_function(), CNN_model.optimizer())
        saver.save(sess, 'ckpt/dpcnnckpt', global_step=i+1)
        print("Loss: ", l)
    sess.close()

    pass
