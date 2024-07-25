import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model, Sequential, load_model, save_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import SGD, Adam
from keras.datasets import mnist
from sklearn.model_selection import KFold
import os
from keras.models import load_model
from numpy import mean, std
from Mymodal import Mymodal
from AlexNet import AlexNet
from VGG import VGG
from GoogleNet import GoogleNet

print('Choose models:')
print('1. Mymodal')
print('2. AlexNet')
print('3. VGG')
print('4. GoogleNet')
choice = int(input('Choose models: '))
while choice < 1 or choice > 4:
    choice = int(input('Choose models again: '))
if choice == 1:
    model_name = Mymodal()
    filename = 'Mymodal_modal.keras'
elif choice == 2:
    model_name = AlexNet()
    filename = 'AlexNet_modal.keras'
elif choice == 3:
    model_name = VGG()
    filename = 'VGG_modal.keras'
elif choice == 4:
    model_name = GoogleNet()
    filename = 'GoogleNet_modal.keras'

#load train and test dataset
def load_dataset():
    (trainX, trainY), (testX, testY) = mnist.load_data()
    #reshape dataset to have a signle channel
    trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
    testX = testX.reshape((testX.shape[0], 28, 28, 1))
    #one hot encode target values
    trainY = to_categorical(trainY)
    testY = to_categorical(testY)
    return trainX, trainY, testX, testY

#prepare pixel data
def prep_pixels(train, test):
    #convert from integers to floats
    train_norm = train.astype('float32')
    test_norm = test.astype('float32')
    #normalize to range 0-1
    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0
    return train_norm, test_norm

def define_model():
    model = model_name
    opt = Adam()
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

#evaluate Model
def evaluate_model(dataX, dataY, n_folds=5):
    scores, histories = list(), list()
    #prepare cross validation
    kfold = KFold(n_folds, shuffle=True, random_state=1)
    #enumerate splits
    for train_ix, test_ix in kfold.split(dataX):
        #define model
        model = define_model()
        #select rows for train and test
        trainX, trainY, testX, testY = dataX[train_ix], dataY[train_ix], dataX[test_ix], dataY[test_ix]
        #fit model
        history = model.fit(trainX, trainY, epochs=10, batch_size=32, validation_data=(testX, testY), verbose=0)
        #evaluate model
        _, acc = model.evaluate(testX, testY, verbose=0)
        print(f'Accuracy: {acc * 100}')
        #append scores and histories
        scores.append(acc)
        histories.append(history)
    return scores, histories

def summarize_performance(scores):
    #print summary
    print(f'Accuracy: mean={mean(scores)*100:.3f} std={std(scores)*100:.3f}, n={len(scores)}')
    #box and whisker plots of results
    plt.boxplot(scores)
    plt.show()

#present result
#plot diagnostic learning curves
def summarize_diagnostics(histories):
    for i in range(len(histories)):
        #plot loss
        plt.subplot(1, 2, 1)
        plt.title('Cross Entropy Loss')
        plt.plot(histories[i].history['loss'], color='blue', label='train')
        plt.plot(histories[i].history['val_loss'], color='orange', label='test')
        #plot accuracy
        plt.subplot(1, 2, 2)
        plt.title('Classification Accuracy')
        plt.plot(histories[i].history['accuracy'], color='blue', label='train')
        plt.plot(histories[i].history['val_accuracy'], color='orange', label='test')
    plt.show()

def recreate_model():
    return model_name

#run test harness for evaluating a model
def run_test_harness():
    #load dataset\
    trainX, trainY, testX, testY = load_dataset()
    #prepare pixel data
    trainX, testX = prep_pixels(trainX, testX)
    #load model
    if os.path.isfile(filename):
        model = load_model(filename, custom_objects={f'{model_name}': recreate_model()})
        # Compile model with the desired optimizer and loss function
        opt = Adam()
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    else:
        #define model
        model = define_model()
    #evaluate model
    scores, histories = evaluate_model(trainX, trainY)
    #learning curves
    summarize_diagnostics(histories)
    #summarize estimated performance
    summarize_performance(scores)
    #fit model
    model.fit(trainX, trainY, epochs=10, batch_size=32, verbose=0)
    #save model
    model.save(filename)

#run test harness
run_test_harness()