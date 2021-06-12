from flask import Flask, request, json
from flask_cors import CORS, cross_origin
import numpy as np
import cv2 as cv
from PIL import Image

app = Flask(__name__)

CORS(app)


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

    # print(image)

    img = np.zeros([256, 256, 4])
    # img = np.array(Image.open('canvas.png'))
    # img2 = np.imre
    img.fill(255)

    print(img)

    init_index = 0

    for i in range(len(img)):
        for j in range(len(img[i])):
            for k in range(len(img[i][j]) - 1):
                img[i][j][k] = image_array[init_index]
            init_index += 1

    print(len(image_array))
    print(init_index)
    print(img)
    # cv.imshow('win', img)
    cv.imwrite('i1.jpg', img)

    return 'sd'


if __name__ == '__main__':
    app.run()
