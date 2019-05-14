import numpy as np
import gzip
import tarfile
import os
# from utils.raw_data import process_image

from tqdm import tqdm

import tensorflow as tf
import tensorflow.python.keras as keras

# from DPCNN import DPCNN 
from model513 import DPCNN



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
    
    rankings = np.load("npy/ranking0509_4_3_70.npy")
    index2name = np.load("npy/index2name.npy", allow_pickle=True).item()
    rankingpl = tf.placeholder(dtype=tf.int32, shape=[None, None])

    vgg_feature_len = 512

    batchsz = 1
    CNN_model = DPCNN(tarpath, rankings, index2name, batchSz=batchsz)
    
    # CNN_model.predict("all_souls_000000.jpg")
    
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    saver = tf.train.Saver(max_to_keep=2)
    init = tf.global_variables_initializer()
    writer = tf.summary.FileWriter("logs/", sess.graph)
    sess.run(init)
    
    saver.save(sess, 'ckpt/dpcnnckpt')
    feed_dict = {rankingpl: rankings}
    
    for i in range(5):
        print("\n========================= epoch being processed:{} =========================".format(i))
        for batch in tqdm(
            range(len(tar.getmembers())//batchsz),
                desc="Procesing Batches", ascii=True):
            Ia_input, Ic_input, If_input, rank_cf = CNN_model.generateImage_acf(batch)
            feed_dict = {CNN_model.Ia_place:Ia_input, CNN_model.Ic_place:Ic_input, CNN_model.If_place:If_input, CNN_model.rank_c_f:rank_cf}
            l, _ = sess.run([CNN_model.loss, CNN_model.train], feed_dict=feed_dict)
            print("loss: ", l)
            if batch%40==0:
                saver.save(sess, 'ckpt/dpcnnckpt', global_step=batch+1)
        
        
        




    for i in range(10):
        print(tf.trainable_variables())
        print("\n========================= epoch being processed:{} =========================".format(i))
        # l, _ = sess.run(CNN_model.loss_function(), CNN_model.optimizer())
        for batch in tqdm(
            range(len(tar.getmembers())//batchsz),
                desc="Procesing Batches", ascii=True):
            _ = sess.run(CNN_model.batch_train(batch))
            print("+1")

        saver.save(sess, 'ckpt/dpcnnckpt', global_step=i+1)
        print("Loss: ", l)
    sess.close()
    pass
