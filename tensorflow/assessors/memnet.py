from __future__ import print_function
import tensorflow as tf
import numpy as np
import scipy.io
import os
import math
import pickle


# The is a tensorflow model translated from a Caffe model.
# Note that I left out any dropout layers
# The tensorflow image resizing yields slightly different results than the resizing used in the original code.
# http://memorability.csail.mit.edu/
# Source for the original model:
# Understanding and Predicting Image Memorability at a Large Scale
# A. Khosla, A. S. Raju, A. Torralba and A. Oliva
# International Conference on Computer Vision (ICCV), 2015
# DOI 10.1109/ICCV.2015.275

__all__ = ['MemNet', 'memnet']

class MemNet():
    def __init__(self):
        pass

    # Preprocessing
    #----------------
    def get_mu(self, mean_image_file):
        if os.path.splitext(mean_image_file)[1] == ".npy":
            mu = np.load(mean_image_file)
        elif os.path.splitext(mean_image_file)[1] == ".mat":
            mu = scipy.io.loadmat(mean_image_file)
            mu = mu["image_mean"]
        if mu.shape[2] != 3:
            mu = mu.transpose((1, 2, 0))

        return mu

    def tencrop(self, myinput=None):
        i = tf.constant(0)
        crops_list = tf.constant(0, shape=[10, 3, 227, 227], dtype=tf.float32)

        def cond(batch, crops_list, k):
            return tf.less(k, tf.shape(batch)[0])

        def body(batch, crops_list, k):
            img = batch[k, :, :, :]
            cropped_size = tf.constant(227)
            indices = [0, 256 - 227]  # hard coded

            for i in indices:
                for j in indices:
                    temp_img = img[i:i + cropped_size, j:j + cropped_size, :]
                    crops_list = tf.concat([crops_list, [tf.transpose(temp_img, (2, 0, 1))]], 0)
                    crops_list = tf.concat([crops_list, [tf.reverse(crops_list[-1], tf.constant([2]))]], 0)
                    center = int(math.floor(indices[1] / 2) + 1)
            crops_list = tf.concat(
                [crops_list,
                 [tf.transpose(img[center:center + cropped_size, center:center + cropped_size, :], (2, 0, 1))]],
                0)
            crops_list = tf.concat([crops_list, [tf.reverse(crops_list[-1], tf.constant([2]))]], 0)

            return batch, crops_list, k + 1

        _, crops_list, _ = tf.while_loop(cond, body, [myinput, crops_list, i],
                                         shape_invariants=[tf.TensorShape([None, 256, 256, 3]),
                                                           tf.TensorShape([None, 3, 227, 227]), i.get_shape()])

        return (crops_list[10:, :, :, :])  # first 10 are dummies

    def channels_last(self, myinput=None):
        channel_dim = [i for i in range(1, 4) if myinput.get_shape().as_list()[i] == 3]
        if len(channel_dim) != 1:
            raise ValueError("Error during memnet preprocessing. Couldn't identify the channel dimension")
        else:
            channel_dim = channel_dim[0]
            myinput = tf.transpose(myinput,[i for i in range(4) if i != channel_dim]+ [channel_dim])
        return myinput

    def memnet_preprocess(self, myinput=None):
        myinput_channels_last = self.channels_last(myinput)
        mu = self.get_mu(os.path.join(os.path.dirname(os.path.realpath(__file__)),"memnet_mean.mat"))
        myinput_BGR = tf.reverse(myinput_channels_last, [3])
        myinput_256 = tf.image.resize_images(myinput_BGR, tf.constant([256, 256]))  # something is off here
        myinput_norm = tf.math.subtract(myinput_256, tf.constant(mu))
        myinput_crops = self.tencrop(myinput_norm)

        return (myinput_crops)

    # Forward pass
    #----------------
    def get_weights(self,weights_path):
        with open(weights_path, 'rb') as f:
            state_dict = pickle.load(f, encoding='latin1')
        return (state_dict)

    def memnet_fn(self, myinput=None):
        """Model function for CNN."""

        paddings_2 = tf.constant([[0, 0], [0, 0], [2, 2], [2, 2]])
        paddings_1 = tf.constant([[0, 0], [0, 0], [1, 1], [1, 1]])

        state_dict = self.get_weights(os.path.join(os.path.dirname(os.path.realpath(__file__)),"memnet_state_dict.p"))

        # Convolutional Layer #1
        conv1 = tf.layers.conv2d(inputs=myinput,
                                 data_format="channels_first",
                                 filters=96,
                                 kernel_size=[11, 11],
                                 strides=[4, 4],
                                 padding="valid",
                                 activation=tf.nn.relu,
                                 kernel_initializer=tf.constant_initializer(
                                     np.transpose(state_dict["conv1.weight"].numpy(), [2, 3, 1, 0])),
                                 bias_initializer=tf.constant_initializer(state_dict["conv1.bias"].numpy()))

        # Pooling Layer #1
        pool1 = tf.layers.max_pooling2d(inputs=conv1,
                                        pool_size=3, strides=2,
                                        data_format="channels_first")

        # Norm Layer #1
        norm1 = tf.transpose(
            tf.nn.local_response_normalization(tf.transpose(pool1, [0, 2, 3, 1]), depth_radius=2, alpha=0.00002,
                                               beta=0.75,
                                               bias=1), [0, 3, 1, 2])
        norm1_padd = tf.pad(norm1, paddings_2)
        norm1_padd_group1 = norm1_padd[:, tf.constant(0):tf.constant(48), :, :]
        norm1_padd_group2 = norm1_padd[:, tf.constant(48):tf.constant(96), :, :]

        # Convolutional Layer #2
        conv2_group1 = tf.layers.conv2d(inputs=norm1_padd_group1,
                                        data_format="channels_first",
                                        filters=128,
                                        kernel_size=[5, 5],
                                        strides=[1, 1],
                                        padding="valid",
                                        activation=tf.nn.relu,
                                        kernel_initializer=tf.constant_initializer(
                                            np.transpose(state_dict["conv2.weight"].numpy()[0:128, :, :, :],
                                                         [2, 3, 1, 0])),
                                        bias_initializer=tf.constant_initializer(
                                            state_dict["conv2.bias"].numpy()[0:128]))
        conv2_group2 = tf.layers.conv2d(inputs=norm1_padd_group2,
                                        data_format="channels_first",
                                        filters=128,
                                        kernel_size=[5, 5],
                                        strides=[1, 1],
                                        padding="valid",
                                        activation=tf.nn.relu,
                                        kernel_initializer=tf.constant_initializer(
                                            np.transpose(state_dict["conv2.weight"].numpy()[128:256, :, :, :],
                                                         [2, 3, 1, 0])),
                                        bias_initializer=tf.constant_initializer(
                                            state_dict["conv2.bias"].numpy()[128:256]))
        conv2 = tf.concat([conv2_group1, conv2_group2], axis=1)

        # Pooling Layer #26
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=3, strides=2, data_format="channels_first")

        # Norm Layer #2
        norm2 = tf.transpose(
            tf.nn.local_response_normalization(tf.transpose(pool2, [0, 2, 3, 1]), depth_radius=2, alpha=0.00002,
                                               beta=0.75,
                                               bias=1), [0, 3, 1, 2])
        norm2_padded = tf.pad(norm2, paddings_1)

        # Convolutional Layer #3
        conv3 = tf.layers.conv2d(inputs=norm2_padded,
                                 data_format="channels_first",
                                 filters=384,
                                 kernel_size=[3, 3],
                                 strides=[1, 1],
                                 padding="valid",
                                 activation=tf.nn.relu,
                                 kernel_initializer=tf.constant_initializer(
                                     np.transpose(state_dict["conv3.weight"].numpy(), [2, 3, 1, 0])),
                                 bias_initializer=tf.constant_initializer(state_dict["conv3.bias"].numpy()))
        conv3_padded = tf.pad(conv3, paddings_1)
        conv3_padded_group1 = conv3_padded[:, tf.constant(0):tf.constant(192), :, :]
        conv3_padded_group2 = conv3_padded[:, tf.constant(192):tf.constant(384), :, :]

        # Convolutional Layer #4
        conv4_group1 = tf.layers.conv2d(inputs=conv3_padded_group1,
                                        data_format="channels_first",
                                        filters=192,
                                        kernel_size=[3, 3],
                                        strides=[1, 1],
                                        padding="valid",
                                        activation=tf.nn.relu,
                                        kernel_initializer=tf.constant_initializer(
                                            np.transpose(state_dict["conv4.weight"].numpy()[0:192, :, :, :],
                                                         [2, 3, 1, 0])),
                                        bias_initializer=tf.constant_initializer(
                                            state_dict["conv4.bias"].numpy()[0:192]))
        conv4_group2 = tf.layers.conv2d(inputs=conv3_padded_group2,
                                        data_format="channels_first",
                                        filters=192,
                                        kernel_size=[3, 3],
                                        strides=[1, 1],
                                        padding="valid",
                                        activation=tf.nn.relu,
                                        kernel_initializer=tf.constant_initializer(
                                            np.transpose(state_dict["conv4.weight"].numpy()[192:384, :, :, :],
                                                         [2, 3, 1, 0])),
                                        bias_initializer=tf.constant_initializer(
                                            state_dict["conv4.bias"].numpy()[192:384]))
        conv4_padded_group1 = tf.pad(conv4_group1, paddings_1)
        conv4_padded_group2 = tf.pad(conv4_group2, paddings_1)

        # Convolutional Layer #5
        conv5_group1 = tf.layers.conv2d(inputs=conv4_padded_group1,
                                        data_format="channels_first",
                                        filters=128,
                                        kernel_size=[3, 3],
                                        strides=[1, 1],
                                        padding="valid",
                                        activation=tf.nn.relu,
                                        kernel_initializer=tf.constant_initializer(
                                            np.transpose(state_dict["conv5.weight"].numpy()[0:128, :, :, :],
                                                         [2, 3, 1, 0])),
                                        bias_initializer=tf.constant_initializer(
                                            state_dict["conv5.bias"].numpy()[0:128]))
        conv5_group2 = tf.layers.conv2d(inputs=conv4_padded_group2,
                                        data_format="channels_first",
                                        filters=128,
                                        kernel_size=[3, 3],
                                        strides=[1, 1],
                                        padding="valid",
                                        activation=tf.nn.relu,
                                        kernel_initializer=tf.constant_initializer(
                                            np.transpose(state_dict["conv5.weight"].numpy()[128:256, :, :, :],
                                                         [2, 3, 1, 0])),
                                        bias_initializer=tf.constant_initializer(
                                            state_dict["conv5.bias"].numpy()[128:256]))
        conv5 = tf.concat([conv5_group1, conv5_group2], axis=1)

        # Pooling Layer #5
        pool5 = tf.layers.max_pooling2d(inputs=conv5, pool_size=3, strides=2, data_format="channels_first")

        # Dense #6
        pool5_flat = tf.layers.Flatten(data_format="channels_last")(pool5)
        fc6 = tf.layers.dense(inputs=pool5_flat,
                              units=4096,
                              activation=tf.nn.relu,
                              kernel_initializer=tf.constant_initializer(
                                  np.transpose(state_dict["fc6.weight"].numpy(), [1, 0])),
                              bias_initializer=tf.constant_initializer(state_dict["fc6.bias"].numpy()))

        # Dense #7
        fc7 = tf.layers.dense(inputs=fc6,
                              units=4096,
                              activation=tf.nn.relu,
                              kernel_initializer=tf.constant_initializer(
                                  np.transpose(state_dict["fc7.weight"].numpy(), [1, 0])),
                              bias_initializer=tf.constant_initializer(state_dict["fc7.bias"].numpy()))

        # Dense #8
        fc8 = tf.layers.dense(inputs=fc7,
                              units=1,
                              activation=None,
                              kernel_initializer=tf.constant_initializer(
                                  np.transpose(state_dict["fc8.weight"].numpy(), [1, 0])),
                              bias_initializer=tf.constant_initializer(state_dict["fc8.bias"].numpy()))
        fc8_mean = tf.math.reduce_mean(fc8, 0)

        # Average per 10 fc8 values
        fc8_expand = tf.expand_dims(fc8, 0)
        fc8_mean = tf.layers.average_pooling1d(inputs=fc8_expand, pool_size=[10], strides=[
            10], padding="valid", data_format="channels_last")
        fc8_mean_squeeze = tf.squeeze(fc8_mean, axis=[0])

        # Linear transformation settings
        mean_pred = tf.constant(0.7626, tf.float32)
        mean_pred_add = tf.constant(0.65, tf.float32)
        pred_rescale = tf.constant(2, tf.float32)
        pred_min = tf.constant(0, tf.float32)
        pred_max = tf.constant(1, tf.float32)

        # Final memorability score
        memorability = tf.math.minimum(tf.math.maximum((fc8_mean_squeeze - mean_pred) *
                                                       pred_rescale + mean_pred_add, pred_min), pred_max)

        return memorability

def memnet(image):
    model = MemNet()
    return(model.memnet_fn(model.memnet_preprocess(image)))