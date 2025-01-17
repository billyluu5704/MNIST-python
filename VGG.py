import numpy as np
import tensorflow as tf
from keras.layers import Dense, Flatten, Conv2D, Input, Dropout, MaxPooling2D
from keras.models import Model, Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist

class VGG(Model):
    def __init__(self, **kwargs):
        super(VGG, self).__init__(**kwargs)
        self.conv_block1 = self.vgg_block(32, 2)
        self.conv_block2 = self.vgg_block(64, 2)
        self.conv_block3 = self.vgg_block(128, 4)
        self.flatten = Flatten()
        #Fully connected layer
        self.dense1 = Dense(128, activation='relu')
        self.dropout = Dropout(0.5) #ideal value for drop rate is 0.2<x<0.5
        self.dense2 = Dense(64, activation='relu')
        self.dropout = Dropout(0.5)
        self.dense3 = Dense(10, activation='softmax') #10 outputs from 0-9
    def vgg_block(self, n_filters, n_conv):
        block = Sequential()
        #add convolutional layers
        for _ in range(n_conv):
            block.add(Conv2D(n_filters, (3, 3), padding='same', activation='relu'))
        #add max pooling layer
        block.add(MaxPooling2D((2,2), strides=(2,2)))
        return block
    def call(self, x):     
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout(x)
        x = self.dense2(x)
        x = self.dropout(x)
        return self.dense3(x)