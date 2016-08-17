import numpy as np
import mxnet as mx
import os
from skimage import io, transform

def PreprocessImage(path):
    # load image
    img = io.imread(path)
    print("Original Image Shape: ", img.shape)
    # we crop image from center
    short_egde = min(img.shape[:2])
    yy = int((img.shape[0] - short_egde) / 2)
    xx = int((img.shape[1] - short_egde) / 2)
    crop_img = img[yy : yy + short_egde, xx : xx + short_egde]
    # resize to 299, 299
    resized_img = transform.resize(crop_img, (299, 299))
    # convert to numpy.ndarray
    sample = np.asarray(resized_img) * 256
    # swap axes to make image from (299, 299, 3) to (3, 299, 299)
    sample = np.swapaxes(sample, 0, 2)
    sample = np.swapaxes(sample, 1, 2)
    # sub mean
    normed_img = sample - 128.
    normed_img /= 128.

    return np.reshape(normed_img, (1, 3, 299, 299))

# extract all image features under the given path
def extract_all_features(path):
    image_names = os.listdir(path)
    image_feats = []
    # these image names order shuold correspond to the question
    for image_name in image_names:
        feature = extract_feature(image_name)
        if len(feature) == 0:
            image_feats.append(None)
        else:
            image_feats.append(feature)
    
    return image_feats

# path is the image's path
def extract_feature(path, cnn_model):
    
    image = PreprocessImage(path)
    flatten_output = cnn_model.predict(image)
    feat = flatten_output[0]
    return np.reshape(feat, (len(feat), 1))