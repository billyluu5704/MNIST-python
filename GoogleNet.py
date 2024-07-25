import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Input, Dropout, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import concatenate
from tensorflow.keras.models import Model, Sequential, load_model, save_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import os

class Inception(layers.Layer):
    def __init__(self, filters, **kwargs):
        super(Inception, self).__init__(**kwargs)
        f1, f2_in, f2_out, f3_in, f3_out, f4 = filters
        self.conv1 = Conv2D(f1, (1, 1), padding='same', activation='relu')
        self.conv3_1 = Conv2D(f2_in, (1,1), padding='same', activation='relu')
        self.conv3_2 = Conv2D(f2_out, (3, 3), padding='same', activation='relu')
        self.conv5_1 = Conv2D(f3_in, (1,1), padding='same', activation='relu')
        self.conv5_2 = Conv2D(f3_out, (5,5), padding='same', activation='relu')
        self.pool = MaxPooling2D((3,3), strides=(1,1), padding='same')
        self.conv1 = Conv2D(f4, (1,1), padding='same', activation='relu')
    def call(self,x):
        conv1 = self.conv1(x)
        conv3_1 = self.conv3_1(x)
        conv3_2 = self.conv3_2(conv3_1)
        conv5_1 = self.conv5_1(x)
        conv5_2 = self.conv5_2(conv5_1)
        pool = self.pool(x)
        conv1_1 = self.conv1(pool)
        layer_out = concatenate([conv1, conv3_2, conv5_2, conv1_1], axis=-1)
        return layer_out

class GoogleNet(Model):
    def __init__(self, **kwargs):
        super(GoogleNet, self).__init__(**kwargs)
        self.conv7 = Conv2D(64, (7,7), strides=(2,2), padding='same', activation='relu')
        self.max_pool = MaxPooling2D((3,3), strides=(2,2), padding='same')
        self.conv3 = Conv2D(192, (3,3), strides=(1,1), padding='same', activation='relu')
        self.avg_pool = AveragePooling2D((7,7), strides=(1,1), padding='same')
        #Inception blocks
        self.inception3a = Inception([64, 96, 128, 16, 32, 32])
        self.inception3b = Inception([128, 128, 192, 32, 96, 64])
        self.inception4a = Inception([192, 96, 208, 16, 48, 64])
        self.inception4b = Inception([160, 112, 224, 24, 64, 64])
        self.inception4c = Inception([128, 128, 256, 24, 64, 64])
        self.inception4d = Inception([112, 144, 288, 32, 64, 64])
        self.inception4e = Inception([256, 160, 320, 32, 128, 128])
        self.inception5a = Inception([256, 160, 320, 32, 128, 128])
        self.inception5b = Inception([384, 192, 384, 48, 128, 128])
        #Fully connected layer
        self.flatten = Flatten()
        self.dense1 = Dense(128, activation='linear')
        self.dropout = Dropout(0.4) #ideal value for drop rate is 0.2<x<0.5
        self.dense2 = Dense(10, activation='softmax') #10 outputs from 0-9
    def call(self, x):
        x = self.conv7(x)
        x = self.max_pool(x)
        x = self.conv3(x)
        x = self.max_pool(x)
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.max_pool(x)
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.max_pool(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout(x)
        return self.dense2(x)



