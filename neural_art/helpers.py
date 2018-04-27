import argparse
import scipy.misc
import numpy as np


def imread(path):
    return scipy.misc.imread(path).astype(np.float)  # returns RGB format


def imsave(path, img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    scipy.misc.imsave(path, img)


def imgpreprocess(image, vgg19_mean):
    image = image[np.newaxis,:,:,:]
    return image - vgg19_mean


def imgunprocess(image, vgg19_mean):
    temp = image + vgg19_mean
    return temp[0]


# function to convert 2D greyscale to 3D RGB
def to_rgb(im):
    w, h = im.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = im
    ret[:, :, 1] = im
    ret[:, :, 2] = im
    return ret


def get_args_from_command_line():
    parser = argparse.ArgumentParser(epilog=__doc__)

    parser.add_argument("-c", "--content_img", type=str, default=None,
                        help='Path to image used as content')
    parser.add_argument("-s", "--style_img", type=str, default=None,
                        help='Path to image used as style')

    return parser.parse_args()
