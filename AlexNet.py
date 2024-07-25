import numpy as np
import tensorflow as tf
from keras.layers import Dense, Flatten, Conv2D, Input, Dropout, MaxPooling2D
from keras.models import Model

class AlexNet(Model):
    def __init__(self, **kwargs):
        super(AlexNet, self).__init__(**kwargs)
        #First Layer
        self.conv1 = Conv2D(32, kernel_size=(3,3), padding='valid', activation='relu', input_shape=(28,28,1)) #start with small value like 32 for number of filters
        self.maxpool1 = MaxPooling2D(pool_size=(2,2))
        #Second Layer
        self.conv2 = Conv2D(64, kernel_size=(3,3), padding='valid', activation='relu')
        self.maxpool2 = MaxPooling2D(pool_size=(2,2))
        #third layer
        self.conv3 = Conv2D(128, kernel_size=(3,3), padding='valid', activation='relu')
        #layer between convolutional and fully connect
        self.flatten = Flatten()
        #Fully connected layer
        self.dense1 = Dense(128, activation='relu')
        self.dropout = Dropout(0.5) #ideal value for drop rate is 0.2<x<0.5
        self.dense2 = Dense(64, activation='relu')
        self.dropout = Dropout(0.5)
        self.dense3 = Dense(10, activation='softmax') #10 outputs from 0-9
    def call(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout(x)
        x = self.dense2(x)
        x = self.dropout(x)
        return self.dense3(x)
    def get_config(self):
        config = super(AlexNet, self).get_config()
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

