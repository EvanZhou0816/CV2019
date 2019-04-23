import tensorflow as tf
import keras
import numpy as np
import gzip
import tarfile


class DPCNN():
    def __init__(self,inputs,output_size,learning_rate,drop_out,hidden_layers,epochs,batchSz):

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

        # # computation graph construction
        # self.construct()

        self.prediction = self.forward_pass()
        self.loss = self.loss_function()
        self.optimize = self.optimizer()
        self.accuracy = self.accuracy_function()
        pass

    def forward_pass(self):
        #parameters unsettled
        image = tf.reshape(self.image, [50, 28, 28, 1])
        filter_1 = tf.Variable(tf.truncated_normal([4, 4, 1, 16], stddev=0.1))
        b1 = tf.Variable(tf.truncated_normal(shape=[16], stddev=0.1))
        conv_1 = tf.nn.conv2d(image, filter_1, [1, 2, 2, 1], "SAME") + b1
        conv_1 = tf.nn.relu(conv_1)
        pool_1 = tf.nn.max_pool(conv_1, ksize=[1,2,2,1], strides = [1,2,2,1], padding='VALID')


        filter_2 = tf.Variable(tf.truncated_normal([2, 2, 16, 64], stddev=0.1))
        b2 = tf.Variable(tf.truncated_normal(shape=[64], stddev=0.1))
        conv_2 = tf.nn.conv2d(pool_1, filter_2, [1, 2, 2, 1], "SAME") + b2
        conv_2 = tf.nn.relu(conv_2)
        pool_2 = tf.nn.max_pool(conv_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        pool_2 = tf.reshape(pool_2, [50, 256])

        w_fc1 = tf.Variable(tf.truncated_normal([256, 100], stddev=0.1))
        b_fc1 = tf.Variable(tf.constant(0.1, shape=[100]))
        h_fc1 = tf.nn.relu(tf.matmul(tf.reshape(pool_2, [-1, 256]), w_fc1) + b_fc1)

        w_fc2 = tf.Variable(tf.truncated_normal([100, 10], stddev=0.1))
        b_fc2 = tf.Variable(tf.constant(0.1, shape=[10]))

        top = tf.nn.dropout(h_fc1, self.keep_prob)

        logits = tf.matmul(top, w_fc2) + b_fc2

        probabilities = tf.nn.softmax(logits)
        return probabilities

    def loss_function(self):
        pass

    def accuracy_function(self):
        pass

    def construct(self):
        pass
    def optimizer(self):
        train = tf.train.GradientDescentOptimizer(0.01).minimize(self.loss)
        return train

    def train(self,verbose=0):
        pass
    def predict(self,inputs,verbose=0):
        pass
    def test(self,test_inputs,test_labels,verbose=0):
        pass

def main():
    with open("oxbuild_images.tgz",'rb') as f:
        print(np.array(f).shape)
    # with open("oxbuild_images.tgz", 'rb') as f:
    #     buf = gzip.GzipFile(fileobj=f).read(5051 * 28 * 28)
    #     train_inputs = np.frombuffer(buf, dtype='uint8', offset=16).reshape(5050, 28 * 28) #all images are resized with the longer side having 600 pixels.
    #     train_inputs = train_inputs / 255

    #     batchSz = 40
    #     img = tf.placeholder(tf.float32, [batchSz, 784])
    #     # keep_prb = tf.placeholder(tf.float32)

    #     dpcnn_model = DPCNN(inputs=train_inputs, learning_rate=0.01)

    #     sess = tf.Session()
    #     sess.run(tf.global_variables_initializer())
    #     opt = dpcnn_model.optimizer()
    #     acc = dpcnn_model.accuracy_function()
    #     for i in range(101):
    #         imgs, anss = mnist.train.next_batch(batchSz)
    #         sess.run(opt, feed_dict={img: imgs})
    #     return

    #     sumAcc = 0
    #     for i in range(1000):
    #         imgs, anss = mnist.test.next_batch(batchSz)
    #         sumAcc += sess.run(acc, feed_dict={img: imgs, ans: anss, keep_prb: 1})
    #     print("Test Accuracy: %r" % (sumAcc / 1000))
    #     return
