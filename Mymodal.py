import numpy as np
import tensorflow as tf
from keras.layers import Dense, Flatten, Conv2D, Input, Dropout, MaxPooling2D
from keras.models import Model
from keras.datasets import mnist
from keras.optimizers import Adam

class Mymodal(Model):
    def __init__(self, **kwargs):
        super(Mymodal, self).__init__(**kwargs)
        self.conv1 = Conv2D(32, 3, activation='relu')
        self.flatten = Flatten()
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(10)
    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)