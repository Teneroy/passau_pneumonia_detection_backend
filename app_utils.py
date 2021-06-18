import numpy as np


def fill_nparray_from_data(nparr, nparr_normalized, data):
    init_index = 0
    for i in range(len(nparr)):
        for j in range(len(nparr[i])):
            for k in range(len(nparr[i][j])):
                nparr_normalized[i][j][k] = float(data[init_index]) / 255.0
                nparr[i][j][k] = data[init_index]
                init_index += 1
    return nparr, nparr_normalized


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])
