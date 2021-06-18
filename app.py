from flask import Flask, request, json
from flask_cors import CORS, cross_origin
import numpy as np
import cv2 as cv
from PIL import Image
from keras.models import load_model
import tensorflow as tf
import eli5
import matplotlib.cm
import keras
import ipython_genutils
from imageio import imread
from skimage.transform import resize
from scipy.stats import wasserstein_distance
from emd import *
from app_utils import *

app = Flask(__name__)

CORS(app)

#model = load_model('model_pneum_co.h5')
model = tf.keras.models.load_model("model_pneum_co.h5")

# print(model)


@app.route('/')
def hello_world():
    return 'Hello World!'


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
    print(img_compare_gray)

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

    # print(img1)
    # print(imgGray)

    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)


    # --VIS PART
    # tf.compat.v1.disable_eager_execution()
    #
    # model_visualize = load_model('model_pneum_co.h5')
    # model_visualize.layers[-1].activation = None
    #
    # visualization = eli5.show_prediction(model_visualize, img, layer="conv2d_6")
    # visualization.save('imgPneum1.png')
    # -- VIS PART END

    #print(visualization.url)

    # print(visualization)
    tf.compat.v1.disable_eager_execution()
    model = tf.keras.models.load_model("model_pneum_co_v1.h5")
    pred = model.predict(img)

    model_visualize = load_model('model_pneum_co_v1.h5')
    model_visualize.layers[-1].activation = None

    #visualization = eli5.show_prediction(model_visualize, img, layer="conv2d_6")

    if pred[0][0] >= 0.5:
        visualization = eli5.show_prediction(model_visualize, img, layer="conv2d_13", targets=[0])
    elif pred[0][0] < 0.5:
        visualization = eli5.show_prediction(model_visualize, img, layer="conv2d_13", targets=[1])

    visualization.save('imgPneum1.png')

    # tf.compat.v1.enable_eager_execution()

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

    # return ''
    # str = ''
    #
    # arr = str.split(" ")
    #
    # idx = 0
    # for i in range(len(img)):
    #     for j in range(len(img[i])):
    #         for z in range(3):
    #             if img[i][j][z] != float(arr[idx]):
    #                 print('WRONG! -> ')
    #                 print(img[i][j][z])
    #                 print(arr[idx])
    #             idx += 1


if __name__ == '__main__':
    app.run()
