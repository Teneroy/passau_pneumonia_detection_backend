
!pip install eli5
from google.colab import files
uploaded = files.upload()

from google.colab import files
uploaded = files.upload()

import sys
import os
import glob
import re
import numpy as np
from PIL import Image
from six.moves import urllib

from keras.utils import np_utils
from skimage import measure
import tensorflow as tf
from keras.preprocessing import image
np.set_printoptions(threshold=sys.maxsize)

model = tf.keras.models.load_model("model_pneum_co_v1.h5")


def model_predict(img, model):
    img = image.load_img(img, target_size=(256, 256), color_mode="rgb") #color_mode="grayscale"
    x = image.img_to_array(img)
    x = x/255    
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)
    return x, preds

x, preds = model_predict('IM-0001-0001.png', model)
#urllib.request.urlretrieve("https://media.springernature.com/lw685/springer-static/image/art%3A10.1186%2Fs12879-020-4869-3/MediaObjects/12879_2020_4869_Fig2_HTML.png", "test.jpg")
#x, preds = model_predict('test.jpg', model)

if preds[0][0] >= 0.5:
  print(str("\nYou are healty by "), preds[0][0]*100)
elif preds[0][0] < 0.5:
  print(str("\nYou have a Pneumonia by "), preds[0][1]*100)

import eli5
import keras
from keras.models import load_model
import matplotlib.cm
tf.compat.v1.disable_eager_execution()

model2 = load_model('model_pneum_co_v1.h5')
model2.layers[-1].activation = None

if preds[0][0] >= 0.5:
  visualization = eli5.show_prediction(model2, x, layer="conv2d_13", targets=[0])
elif preds[0][0] < 0.5:
  visualization = eli5.show_prediction(model2, x, layer="conv2d_13", targets=[1])

visualization.save("visualized_image.png")
Image.open("visualized_image.png") 