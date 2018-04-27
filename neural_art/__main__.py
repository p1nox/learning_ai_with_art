# Source article: http://www.subsubroutine.com/sub-subroutine/2016/11/12/painting-like-van-gogh-with-convolutional-neural-networks
# Source gist: https://gist.github.com/standarderror/855adcadce2e627a7b9a25d6261e03a6#file-20161112_neural_style_tensorflow-py

import os
import time
import math
import scipy.misc
import scipy.io
from sys import stderr
import numpy as np
import tensorflow as tf

from neural_art.helpers import imread, imsave, \
    imgpreprocess, imgunprocess, to_rgb, \
    get_args_from_command_line


## Inputs
args = get_args_from_command_line()
file_content_image = args.content_img
file_style_image = args.style_img

## Parameters
input_noise = 0.1  # proportion noise to apply to content image
weight_style = 2e2

## Layers
layer_content = 'conv4_2'
layers_style = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
layers_style_weights = [0.2, 0.2, 0.2, 0.2, 0.2]

## VGG19 model
path_VGG19 = 'assets/imagenet-vgg-verydeep-19.mat'
# VGG19 mean for standardisation (RGB)
VGG19_mean = np.array([123.68, 116.779, 103.939]).reshape((1, 1, 1, 3))

## Reporting & writing checkpoint images
# NB. the total # of iterations run will be n_checkpoints * n_iterations_checkpoint
n_checkpoints = 10              # number of checkpoints
n_iterations_checkpoint = 10    # learning iterations per checkpoint
path_output = 'assets/outputs'  # directory to write checkpoint images into


### Preprocessing
# create output directory
if not os.path.exists(path_output):
    os.mkdir(path_output)

# read in images
img_content = imread(file_content_image)
img_style = imread(file_style_image)

# convert if greyscale
if len(img_content.shape) == 2:
    img_content = to_rgb(img_content)

if len(img_style.shape) == 2:
    img_style = to_rgb(img_style)

# resize style image to match content
img_style = scipy.misc.imresize(img_style, img_content.shape)

# apply noise to create initial "canvas"
noise = np.random.uniform(
        img_content.mean() - img_content.std(), img_content.mean() + img_content.std(),
        (img_content.shape)).astype('float32')
img_initial = noise * input_noise + img_content * (1 - input_noise)

# preprocess each
img_content = imgpreprocess(img_content, VGG19_mean)
img_style = imgpreprocess(img_style, VGG19_mean)
img_initial = imgpreprocess(img_initial, VGG19_mean)


#### BUILD VGG19 MODEL
## with thanks to http://www.chioka.in/tensorflow-implementation-neural-algorithm-of-artistic-style
VGG19 = scipy.io.loadmat(path_VGG19)
VGG19_layers = VGG19['layers'][0]


# help functions
def _conv2d_relu(prev_layer, n_layer, layer_name):
    # get weights for this layer:
    weights = VGG19_layers[n_layer][0][0][2][0][0]
    W = tf.constant(weights)
    bias = VGG19_layers[n_layer][0][0][2][0][1]
    b = tf.constant(np.reshape(bias, (bias.size)))
    # create a conv2d layer
    conv2d = tf.nn.conv2d(prev_layer, filter=W, strides=[1, 1, 1, 1], padding='SAME') + b
    # add a ReLU function and return
    return tf.nn.relu(conv2d)


def _avgpool(prev_layer):
    return tf.nn.avg_pool(prev_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# Setup network
with tf.Session() as sess:
    a, h, w, d = img_content.shape
    net = {}
    net['input']    = tf.Variable(np.zeros((a, h, w, d), dtype=np.float32))
    net['conv1_1']  = _conv2d_relu(net['input'], 0, 'conv1_1')
    net['conv1_2']  = _conv2d_relu(net['conv1_1'], 2, 'conv1_2')
    net['avgpool1'] = _avgpool(net['conv1_2'])
    net['conv2_1']  = _conv2d_relu(net['avgpool1'], 5, 'conv2_1')
    net['conv2_2']  = _conv2d_relu(net['conv2_1'], 7, 'conv2_2')
    net['avgpool2'] = _avgpool(net['conv2_2'])
    net['conv3_1']  = _conv2d_relu(net['avgpool2'], 10, 'conv3_1')
    net['conv3_2']  = _conv2d_relu(net['conv3_1'], 12, 'conv3_2')
    net['conv3_3']  = _conv2d_relu(net['conv3_2'], 14, 'conv3_3')
    net['conv3_4']  = _conv2d_relu(net['conv3_3'], 16, 'conv3_4')
    net['avgpool3'] = _avgpool(net['conv3_4'])
    net['conv4_1']  = _conv2d_relu(net['avgpool3'], 19, 'conv4_1')
    net['conv4_2']  = _conv2d_relu(net['conv4_1'], 21, 'conv4_2')
    net['conv4_3']  = _conv2d_relu(net['conv4_2'], 23, 'conv4_3')
    net['conv4_4']  = _conv2d_relu(net['conv4_3'], 25, 'conv4_4')
    net['avgpool4'] = _avgpool(net['conv4_4'])
    net['conv5_1']  = _conv2d_relu(net['avgpool4'], 28, 'conv5_1')
    net['conv5_2']  = _conv2d_relu(net['conv5_1'], 30, 'conv5_2')
    net['conv5_3']  = _conv2d_relu(net['conv5_2'], 32, 'conv5_3')
    net['conv5_4']  = _conv2d_relu(net['conv5_3'], 34, 'conv5_4')
    net['avgpool5'] = _avgpool(net['conv5_4'])


### CONTENT LOSS: FUNCTION TO CALCULATE AND INSTANTIATION
# with thanks to https://github.com/cysmith/neural-style-tf

# Recode to be simpler: http://www.chioka.in/tensorflow-implementation-neural-algorithm-of-artistic-style
def content_layer_loss(p, x):
    _, h, w, d = [i.value for i in p.get_shape()]    # d: number of filters; h,w : height, width
    M = h * w
    N = d
    K = 1. / (2. * N**0.5 * M**0.5)
    loss = K * tf.reduce_sum(tf.pow((x - p), 2))
    return loss


with tf.Session() as sess:
    sess.run(net['input'].assign(img_content))
    p = sess.run(net[layer_content])  # Get activation output for content layer
    x = net[layer_content]
    p = tf.convert_to_tensor(p)
    content_loss = content_layer_loss(p, x)


### STYLE LOSS: FUNCTION TO CALCULATE AND INSTANTIATION
def style_layer_loss(a, x):
    _, h, w, d = [i.value for i in a.get_shape()]
    M = h * w
    N = d
    A = gram_matrix(a, M, N)
    G = gram_matrix(x, M, N)
    loss = (1./(4 * N**2 * M**2)) * tf.reduce_sum(tf.pow((G - A), 2))
    return loss


def gram_matrix(x, M, N):
    F = tf.reshape(x, (M, N))
    G = tf.matmul(tf.transpose(F), F)
    return G


with tf.Session() as sess:
    sess.run(net['input'].assign(img_style))
    style_loss = 0.
    # style loss is calculated for each style layer and summed
    for layer, weight in zip(layers_style, layers_style_weights):
        a = sess.run(net[layer])
        x = net[layer]
        a = tf.convert_to_tensor(a)
        style_loss += style_layer_loss(a, x)

### Define loss function and minimise
with tf.Session() as sess:
    # loss function
    L_total = content_loss + weight_style * style_loss

    # instantiate optimiser
    optimizer = tf.contrib.opt.ScipyOptimizerInterface(
      L_total, method='L-BFGS-B',
      options={'maxiter': n_iterations_checkpoint})

    init_op = tf.initialize_all_variables()
    sess.run(init_op)
    sess.run(net['input'].assign(img_initial))
    for i in range(1, n_checkpoints + 1):
        # run optimisation
        optimizer.minimize(sess)

        ## print costs
        stderr.write('Iteration %d/%d\n' % (i*n_iterations_checkpoint, n_checkpoints * n_iterations_checkpoint))
        stderr.write('  content loss: %g\n' % sess.run(content_loss))
        stderr.write('    style loss: %g\n' % sess.run(weight_style * style_loss))
        stderr.write('    total loss: %g\n' % sess.run(L_total))

        ## write image
        img_output = sess.run(net['input'])
        img_output = imgunprocess(img_output, VGG19_mean)
        timestr = time.strftime("%Y%m%d_%H%M%S")
        output_file = path_output+'/'+timestr+'_'+'%s.jpg' % (i*n_iterations_checkpoint)
        imsave(output_file, img_output)
