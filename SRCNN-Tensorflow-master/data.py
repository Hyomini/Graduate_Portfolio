import os
import glob
import numpy as np
from PIL import Image


import tensorflow as tf

FLAGS = tf.app.flags.FLAGS


def preprocess(path, scale):
    """
    Preprocess single image file
      (1) Read original image as YCbCr format (and grayscale as default)
      (2) Normalize
      (3) Apply image file with bicubic interpolation

    Args:
      path: file path of desired file
      input_: image applied bicubic interpolation (low-resolution)
      label_: image with original resolution (high-resolution)
    """
    image = Image.open(path)
    label_ = image.convert("L")
    area = (0,0,32,32)
    label_ = label_.crop(area)

    h, w = label_.size

    temp = label_.resize((int(h/scale), int(w/scale)), resample=Image.BICUBIC)
    input_ = temp.resize((h, w), resample=0)

    return input_, label_


def prepare_data(dataset):
    """
    Args:
      dataset: choose train dataset or test dataset

      For train dataset, output data would be ['.../t1.bmp', '.../t2.bmp', ..., '.../t99.bmp']
    """
    if FLAGS.is_train:
        data_dir = os.path.join(os.sep, (os.path.join(os.getcwd(), dataset)), "291")
        data = glob.glob(os.path.join(data_dir, "*.*"))
        data = data[0:280]
    else:
        data_dir = os.path.join(os.sep, (os.path.join(os.getcwd(), dataset)), "291")
        data = glob.glob(os.path.join(data_dir, "*.*"))
        data = data[281:290]

    return data


def input_setup(config):
    """
    Read image files and make their sub-images and saved them as a h5 file format.
    """

    input_sequence = list()  # input images
    label_sequence = list()  # label images

    # Load data path
    data = prepare_data(dataset="Train")

    if FLAGS.is_train:
        for i in range(len(data)):
            input_, label_ = preprocess(data[i], config.scale)
            input_sequence.append(input_)
            label_sequence.append(label_)

    else:
        for i in range(len(data)):
            input_, label_ = preprocess(data[i], config.scale)
            input_sequence.append(input_)
            label_sequence.append(label_)

    input_images = list()  # input images to numpy array
    label_images = list()  # label images to numpy array
    label_sequence[3].show()
    input_sequence[3].show()
    for i in range(len(input_sequence)):
        temp_input = np.array(input_sequence[i])/255
        temp_label = np.array(label_sequence[i])/255

        input_images.append(temp_input.reshape(32,32,1))
        label_images.append(temp_label.reshape(32,32,1))

    return input_images, label_images

'''
def imsave(image, path):
  return scipy.misc.imsave(path, image)
'''