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


app = Flask(__name__)

CORS(app)

model = load_model('model_pneum_co.h5')


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

    image_array = []
    for val in image:
        image_array.append(image['' + val])

    img = np.zeros([256, 256, 3])
    img_compare = np.zeros([256, 256, 3])

    # print(img)

    init_index = 0

    for i in range(len(img)):
        for j in range(len(img[i])):
            for k in range(len(img[i][j])):
                img[i][j][k] = float(image_array[init_index]) / 255.0
                img_compare[i][j][k] = image_array[init_index]
                init_index += 1

    # print(len(image_array))
    # print(init_index)
    # print(img)
    # cv.imshow('win', img)
    # cv.cvtColor(img, cv.CV_8UC1)
    # cv.imwrite('i1.jpg', img)


    #imgGray = None
    #cv.cvtColor(img_compare, imgGray, cv.COLOR_RGB2GRAY)

    #img1 = imread('i1.jpg', as_gray=True).astype(int)

    #print(img1)
    #print(imgGray)

    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)

    tf.compat.v1.disable_eager_execution()

    model_visualize = load_model('model_pneum_co.h5')
    model_visualize.layers[-1].activation = None

    visualization = eli5.show_prediction(model_visualize, img, layer="conv2d_6")
    visualization.save('imgPneum1.png')

    print(visualization.url)

    # print(visualization)

    # tf.compat.v1.enable_eager_execution()

    if pred[0][0] >= 0.5:
        return json.jsonify(
            {
                "probability": pred[0][0] * 100,
                "existence": False
            }
        )
    elif pred[0][0] < 0.5:
        return json.jsonify(
            {
                "probability": pred[0][1] * 100,
                "existence": True
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
