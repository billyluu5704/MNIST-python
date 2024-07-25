from keras.preprocessing.image import load_img, img_to_array
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import get_custom_objects
import keras
from AlexNet import AlexNet
from VGG import VGG

file = 'VGG_modal.keras'
model_name = VGG()

def recreate_model():
    return model_name

#make prediction
#load and prepare the image
def load_image(filename):
    #load image
    img = load_img(filename, color_mode='grayscale', target_size=(28,28))
    #convert to array
    img = img_to_array(img)
    #reshape into a single sample with 1 channel
    img = img.reshape(1, 28, 28, 1)
    img = img.astype('float32')
    img = img / 255.0
    return img

#load an image and predict the class
def run_example():
    #load image
    img = load_image('images.png')
    #load model
    model = load_model(file, custom_objects={f'{model_name}': recreate_model()})
    #predict the class
    prediction = model.predict(img)
    result = np.argmax(prediction, axis=1)
    print(result[0])

run_example()
