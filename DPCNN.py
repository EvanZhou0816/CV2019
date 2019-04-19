import tensorflow as tf
import keras
import numpy

class DPCNN():
    def __init__(self,inputs,labels,output_size,learning_rate,drop_out,hidden_layers,epochs,batchSz):

        # pre-trained Network(RestNet)
        self.pretrained = keras.applications.ResNet50(weights='imagenet', include_top=False)
        self.inputs = inputs


        # use of data
        




        # Hyper-parameter
        self.learning_rate = learning_rate
        self.output_size = output_size
        self.dropout = drop_out
        self.hidden_layers = hidden_layers
        self.epochs = epochs
        self.batch_size = batchSz

        # computation graph construction
        self.construct()
        pass
    def construct(self):
        pass
    def train(self,verbose=0):
        pass
    def predict(self,inputs,verbose=0):
        pass
    def test(self,test_inputs,test_labels,verbose=0):
        pass