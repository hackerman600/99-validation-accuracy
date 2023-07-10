from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.layers import Flatten,Dense,Conv2D,MaxPooling2D,PReLU,InputLayer,AveragePooling2D
from tensorflow.keras.layers import Layer
from tensorflow.keras import activations
from tensorflow.keras import Input
import matplotlib.pyplot as plt
import keras.layers as layers
import tensorflow as tf
import numpy as np
import keras

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(np.array(x_train).shape, np.array(y_train).shape)
x_test = x_test / 255
x_test = x_test.reshape(-1, 28, 28, 1)
x_train = x_train / 255 
x_train = x_train.reshape(-1, 28, 28, 1)


class CustomDense(Layer):
    def __init__(self, num_units, activation="relu"):
        super(CustomDense, self).__init__()

        self.num_units = num_units

    def build(self, input_shape):  
        ## (32, 784) * (784, 10) + (10)  
        self.weight = self.add_weight(shape=[input_shape[-1], self.num_units])
        self.bais = self.add_weight(shape=[self.num_units])

    def call(self, input):
        #print("input is: ", input)
        y = tf.matmul(input, self.weight) + self.bais
        return activations.relu(y)    


class DenseBlock(Layer):
    def __init__(self, nodes, activation="relu"):
        super(DenseBlock, self).__init__()

        self.init_dense = Dense(nodes, activation = "relu")
        self.scnd_dense = Dense(nodes/3*2, activation = "relu") 

    def call(self, input):
        #print("input is: ", input.shape)
        y = self.init_dense(input)
        y_mean = tf.reduce_sum(y)/y.shape[1]
        scnd = self.scnd_dense(y)
        test = tf.fill([1,scnd.shape[1]], y_mean)
        #print(y.shape, scnd.shape)
        scnd += test
               
        return scnd

         

class CustomConv(Layer):
    def __init__(self, filters,size, activation="relu"):
        super(CustomConv, self).__init__()
        self.conv = Conv2D(filters,size)
      
    def call(self, input):
        #print("input is: ", input)
        out = activations.relu(self.conv(input))
        #print("shape of out is: ", out.shape)
        padded_out =  tf.pad(out, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]])
        #print(padded_out.shape, input.shape)
        new_out = padded_out + input 
        return new_out
    
       
model = Sequential()
model.add(Conv2D(60,(3,3),input_shape = (28,28,1)))
model.add(AveragePooling2D(pool_size=(2, 2)))
model.add(CustomConv(60,(3,3)))
model.add(AveragePooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(DenseBlock(75))
model.add(Dense(10, activation="softmax"))


model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])


model.fit(x_train, y_train, epochs=10,validation_data = (x_test, y_test))
#model.save("/home/kali/Desktop/machine_learning/Neural_networks/basic_non_conv/0.888__2_layers_deeper_higher_epoch/deeper_basic_save.h5")


