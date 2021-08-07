from imageio import imread
from skimage.transform import resize
import numpy as np
from scipy.stats import wasserstein_distance


threshold = 0.00095

canonical_img = "xray4.png"


def emd_comparison(img):
    distance = get_distance(img)
    print('core_img == ' + canonical_img + ' : ' + str(distance))
    return distance <= threshold


def get_distance(img):
    img_src = img
    hist_src = get_histogram(img_src)

    img_in = get_img('./emd_research_data/right/' + canonical_img, norm_exposure=True)
    hist_in = get_histogram(img_in)
    distance = wasserstein_distance(hist_in, hist_src)
    return distance


def get_img(path, norm_size=True, norm_exposure=False):
    """
    Prepare an image for image processing tasks
    """
    # flatten returns a 2d grayscale array
    img = imread(path, as_gray=True).astype(int)
    # resizing returns float vals 0:255; convert to ints for downstream tasks
    return prepare_img(img, norm_exposure=norm_exposure)


def prepare_img(img, norm_size=True, norm_exposure=False):
    """
    Prepare an image for image processing tasks
    """
    # resizing returns float vals 0:255; convert to ints for downstream tasks
    if norm_size:
        img = resize(img, (256, 256), anti_aliasing=True, preserve_range=True)
    if norm_exposure:
        img = normalize_exposure(img)
    return img


def get_histogram(img):
    """
    Get the histogram of an image. For an 8-bit, grayscale image, the
    histogram will be a 256 unit vector in which the nth value indicates
    the percent of the pixels in the image with the given darkness level.
    The histogram's values sum to 1.
    """
    h, w = img.shape
    hist = [0.0] * 256
    for i in range(h):
        for j in range(w):
            hist[img[i, j]] += 1
    return np.array(hist) / (h * w)


def normalize_exposure(img):
    img = img.astype(int)
    hist = get_histogram(img)
    # get the sum of vals accumulated by each position in hist
    cdf = np.array([sum(hist[:i + 1]) for i in range(len(hist))])
    # determine the normalization values for each unit of the cdf
    sk = np.uint8(255 * cdf)
    # normalize each position in the output image
    height, width = img.shape
    normalized = np.zeros_like(img)
    for i in range(0, height):
        for j in range(0, width):
            normalized[i, j] = sk[img[i, j]]
    return normalized.astype(int)
