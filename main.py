import numpy as np
import gzip
import tarfile
import os
from utils.raw_data import process_image
from PIL import Image
import tensorflow as tf
import tensorflow.python.keras as keras
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense, Dropout, Flatten
from tensorflow.python.keras.applications.resnet50 import preprocess_input 

if __name__ == "__main__":
    print(os.getcwd())
    os.chdir('F:/workspace_Python/CV2019')
    tar = tarfile.open("../oxbuild_images.tgz")
    ResNet_model = keras.applications.ResNet50(weights='imagenet', include_top=False,pooling='avg')
    # outputs1 = ResNet_model.get_layer('activation_48').output
    # outputs1 = Dense(4096, activation='softmax')(outputs1)
    # extracter = Model(inputs=ResNet_model.input,outputs=outputs1) 


    for layer in ResNet_model.layers:
        print(layer.name, layer.trainable)
    sess = tf.Session()


    
    # i = 0
    # features_matrix=[0 for i in range(2048)]
    # features_matrix = []
    # for tar_info in tar.getmembers():
    #     f = tar.extractfile(tar_info)
    #     f.read()
    #     res = process_image(f)
    #     res = np.expand_dims(res, axis=0)
    #     res = preprocess_input(res)
    #     features = ResNet_model.predict(res)
    #     # features_reduce = features.squeeze()
    #     print("img: ", i)
    #     # print("feature shape: ",np.shape(features))
    #     if i == 0:
    #         features_matrix = features
    #     else:
    #         features_matrix = np.vstack((features_matrix, features))
    #     print("features matrix: ", np.shape(features_matrix))
    #     i+=1
    # print("image count: ", i)
    # np.save("features_matrix.npy", features_matrix)
        
    pass