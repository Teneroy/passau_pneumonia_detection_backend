#pip install tf-keras-vis

import numpy as np
import cv2 as cv
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib import cm
import matplotlib.cm
from imageio import imread
import ipython_genutils
from skimage.transform import resize
from scipy.stats import wasserstein_distance
from emd import *

from keras.models import load_model
import tensorflow as tf
from tf_keras_vis.utils.scores import CategoricalScore
from tf_keras_vis.saliency import Saliency
import keras
import eli5

from app_utils import *
from flask import Flask, request, json
from flask_cors import CORS, cross_origin

app = Flask(__name__)

CORS(app)

model = tf.keras.models.load_model("model_pneum_co_v1.h5")

@app.route('/api/predictPneumonia', methods=["POST"])
def predict_pneumonia():
    data = json.loads(request.data)
    image = data.get("image")
    size = data.get("size")
    original_width = size.get("width")
    original_height = size.get("height")
    forced = data.get("forced")

    image_array = []
    for val in image:
        image_array.append(image['' + val])

    img = np.zeros([256, 256, 3])
    img_compare = np.zeros([256, 256, 3])

    img_compare, img = fill_nparray_from_data(img, img_compare, image_array)

    img_compare_gray = rgb2gray(img_compare)

    img_a = prepare_img(img_compare_gray, norm_exposure=True)

    is_xray = forced

    if not forced:
        is_xray = emd_comparison(img_a)

    if not is_xray:
        return json.jsonify(
            {
                "probability": 0,
                "existence": False,
                "isXray": False
            }
        )

    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)

    # --VISUALISATION STARTS
   
    def model_modifier(cloned_model):
        cloned_model.layers[-1].activation = tf.keras.activations.linear
        return cloned_model
    
    if preds[0][0] >= 0.5:
      score = CategoricalScore([0])
    elif preds[0][0] < 0.5:
      score = CategoricalScore([1])
    
    smooth_grad = Saliency(model, model_modifier=model_modifier, clone=False)
    smooth_grad_map = smooth_grad(score, seed_input=x[0], smooth_samples=5, smooth_noise=0.0001)
    
    f, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 8))
    ax.set_title("smooth_grad_map", fontsize=16)
    ax.imshow(img[0], interpolation='none')
    ax.imshow(smooth_grad_map[0], cmap='inferno', alpha=0.5, interpolation='none')
    ax.axis('off')
    f.savefig("imgPneum1.png")

    # --VISUALISATION END

    if pred[0][0] >= 0.5:
        return json.jsonify(
            {
                "probability": pred[0][0] * 100,
                "existence": False,
                "isXray": True
            }
        )
    elif pred[0][0] < 0.5:
        return json.jsonify(
            {
                "probability": pred[0][1] * 100,
                "existence": True,
                "isXray": True
            }
        )


if __name__ == '__main__':
    app.run()
