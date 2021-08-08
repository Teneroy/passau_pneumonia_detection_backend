# pip install tf-keras-vis
import os
import threading
import time
import uuid
from os import listdir

import tensorflow as tf
from flask import Flask, request, json
from flask_cors import CORS
from matplotlib import pyplot as plt
from tf_keras_vis.saliency import Saliency
from tf_keras_vis.utils.scores import CategoricalScore

from app_utils import *
from emd import *

app = Flask(__name__)

CORS(app)

model = tf.keras.models.load_model("model_pneum_co_v1.h5")


def cleaning():
    print("Cleaner is waiting")
    while True:
        time.sleep(7200)
        print("Start cleaning")
        for filename in listdir('./static'):
            os.remove('static/' + filename)
        print("Stop cleaning")


def model_modifier(cloned_model):
    cloned_model.layers[-1].activation = tf.keras.activations.linear
    return cloned_model


cleaning_thread = threading.Thread(target=cleaning)
cleaning_thread.start()


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

    score = 0
    if pred[0][0] >= 0.5:
        score = CategoricalScore([0])
    elif pred[0][0] < 0.5:
        score = CategoricalScore([1])

    smooth_grad = Saliency(model, model_modifier=model_modifier, clone=False)
    smooth_grad_map = smooth_grad(score, seed_input=img[0], smooth_samples=5, smooth_noise=0.0001)

    f, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
    ax.set_title("smooth_grad_map", fontsize=16)
    ax.imshow(img[0], interpolation='none')
    ax.imshow(smooth_grad_map[0], cmap='inferno', alpha=0.5, interpolation='none')
    ax.axis('off')

    filename = "static/" + str(uuid.uuid4()) + ".png"
    f.savefig(filename)

    # --VISUALISATION END

    if pred[0][0] >= 0.5:
        return json.jsonify(
            {
                "probability": pred[0][0] * 100,
                "existence": False,
                "isXray": True,
                "visualization": "",
            }
        )
    elif pred[0][0] < 0.5:
        return json.jsonify(
            {
                "probability": pred[0][1] * 100,
                "existence": True,
                "isXray": True,
                "visualization": filename,
            }
        )


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
