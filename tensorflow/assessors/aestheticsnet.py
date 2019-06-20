from __future__ import print_function
import tensorflow as tf
import numpy as np
import scipy.io
import os
import math
import pickle


# The is a tensorflow model translated from a Caffe model.
# Note that I left out any dropout layers
# https://github.com/aimerykong/deepImageAestheticsAnalysis
# Source for the original model:
# Kong, S., Shen, X., Lin, Z., Mech, R., & Fowlkes, C. (2016). Photo aesthetics ranking network with attributes and content adaptation. ArXiv CS, 1606.01621. Retrieved from http://arxiv.org/abs/1606.01621

__all__ = ["AestheticsNet","aestheticsnet"]

class AestheticsNet():
    def __init__(self):
        pass

    # Preprocessing
    #--------------------
    def get_mu(self, mean_image_file):
        mu = np.load(mean_image_file)
        mu = mu[:, 14:241, 14:241]
        return mu

    def channels_first(self, myinput=None):
        channel_dim = [i for i in range(1, 4) if myinput.get_shape().as_list()[i] == 3]
        if len(channel_dim) != 1:
            raise ValueError("Error during memnet preprocessing. Couldn't identify the channel dimension")
        else:
            channel_dim = channel_dim[0]
            myinput = tf.transpose(myinput,[0] + [channel_dim]+[i for i in range(1,4) if i != channel_dim])
        return myinput

    def aestheticsnet_preprocess(self,myinput=None):
        mu = self.get_mu(os.path.join(os.path.dirname(os.path.realpath(__file__)),"mean_AADB_regression_warp256_lore.npy"))
        myinput_BGR = tf.reverse(myinput, [3])
        myinput_227 = tf.image.resize_images(myinput_BGR, tf.constant([227, 227]))  # tensorflow resizing does not work in the exact same way as in the original code
        myinput_channels_first = self.channels_first(myinput_227)
        myinput_norm = tf.math.subtract(myinput_channels_first, tf.constant(mu))
        return (myinput_norm)

    # Forward pass
    #--------------------
    def get_weights(self, weights_path):
        with open(weights_path, 'rb') as f:
            state_dict = pickle.load(f, encoding='latin1')
        return (state_dict)

    def aestheticsnet_fn(self, myinput=None, attribute="aesthetics"):
        """Model function for CNN."""

        paddings_2 = tf.constant([[0, 0], [0, 0], [2, 2], [2, 2]])
        paddings_1 = tf.constant([[0, 0], [0, 0], [1, 1], [1, 1]])

        state_dict = self.get_weights(os.path.join(os.path.dirname(os.path.realpath(__file__)),"aestheticsnet_state_dict.p"))

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
                                        padding="valid",  # Does this have to match pytorch?
                                        # No grouping argument in tf?
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
                              units=512,
                              activation=tf.nn.relu,
                              kernel_initializer=tf.constant_initializer(
                                  np.transpose(state_dict["fc8_new.weight"].numpy(), [1, 0])),
                              bias_initializer=tf.constant_initializer(state_dict["fc8_new.bias"].numpy()))

        # BalancingElement
        fc8_BalancingElement = tf.layers.dense(inputs=fc7,
                                               units=256,
                                               activation=tf.nn.relu,
                                               kernel_initializer=tf.constant_initializer(
                                                   np.transpose(state_dict["fc8_BalancingElement.weight"].numpy(),
                                                                [1, 0])),
                                               bias_initializer=tf.constant_initializer(
                                                   state_dict["fc8_BalancingElement.bias"].numpy())
                                               )
        BalancingElement = tf.layers.dense(inputs=fc8_BalancingElement,
                                           units=1,
                                           activation=None,
                                           kernel_initializer=tf.constant_initializer(
                                               np.transpose(state_dict["fc9_BalancingElement.weight"].numpy(), [1, 0])),
                                           bias_initializer=tf.constant_initializer(
                                               state_dict["fc9_BalancingElement.bias"].numpy())
                                           )
        # ColorHarmony
        fc8_ColorHarmony = tf.layers.dense(inputs=fc7,
                                           units=256,
                                           activation=tf.nn.relu,
                                           kernel_initializer=tf.constant_initializer(
                                               np.transpose(state_dict["fc8_ColorHarmony.weight"].numpy(), [1, 0])),
                                           bias_initializer=tf.constant_initializer(
                                               state_dict["fc8_ColorHarmony.bias"].numpy())
                                           )
        ColorHarmony = tf.layers.dense(inputs=fc8_ColorHarmony,
                                       units=1,
                                       activation=None,
                                       kernel_initializer=tf.constant_initializer(
                                           np.transpose(state_dict["fc9_ColorHarmony.weight"].numpy(), [1, 0])),
                                       bias_initializer=tf.constant_initializer(
                                           state_dict["fc9_ColorHarmony.bias"].numpy())

                                       )

        # Content
        fc8_Content = tf.layers.dense(inputs=fc7,
                                      units=256,
                                      activation=tf.nn.relu,
                                      kernel_initializer=tf.constant_initializer(
                                          np.transpose(state_dict["fc8_Content.weight"].numpy(), [1, 0])),
                                      bias_initializer=tf.constant_initializer(state_dict["fc8_Content.bias"].numpy())
                                      )
        Content = tf.layers.dense(inputs=fc8_Content,
                                  units=1,
                                  activation=None,
                                  kernel_initializer=tf.constant_initializer(
                                      np.transpose(state_dict["fc9_Content.weight"].numpy(), [1, 0])),
                                  bias_initializer=tf.constant_initializer(
                                      state_dict["fc9_Content.bias"].numpy())

                                  )

        # DoF
        fc8_DoF = tf.layers.dense(inputs=fc7,
                                  units=256,
                                  activation=tf.nn.relu,
                                  kernel_initializer=tf.constant_initializer(
                                      np.transpose(state_dict["fc8_DoF.weight"].numpy(), [1, 0])),
                                  bias_initializer=tf.constant_initializer(state_dict["fc8_DoF.bias"].numpy())
                                  )
        DoF = tf.layers.dense(inputs=fc8_DoF,
                              units=1,
                              activation=None,
                              kernel_initializer=tf.constant_initializer(
                                  np.transpose(state_dict["fc9_DoF.weight"].numpy(), [1, 0])),
                              bias_initializer=tf.constant_initializer(
                                  state_dict["fc9_DoF.bias"].numpy())

                              )
        # Light
        fc8_Light = tf.layers.dense(inputs=fc7,
                                    units=256,
                                    activation=tf.nn.relu,
                                    kernel_initializer=tf.constant_initializer(
                                        np.transpose(state_dict["fc8_Light.weight"].numpy(), [1, 0])),
                                    bias_initializer=tf.constant_initializer(state_dict["fc8_Light.bias"].numpy())
                                    )
        Light = tf.layers.dense(inputs=fc8_Light,
                                units=1,
                                activation=None,
                                kernel_initializer=tf.constant_initializer(
                                    np.transpose(state_dict["fc9_Light.weight"].numpy(), [1, 0])),
                                bias_initializer=tf.constant_initializer(
                                    state_dict["fc9_Light.bias"].numpy())

                                )

        # MotionBlur
        fc8_MotionBlur = tf.layers.dense(inputs=fc7,
                                         units=256,
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.constant_initializer(
                                             np.transpose(state_dict["fc8_MotionBlur.weight"].numpy(), [1, 0])),
                                         bias_initializer=tf.constant_initializer(
                                             state_dict["fc8_MotionBlur.bias"].numpy())
                                         )
        MotionBlur = tf.layers.dense(inputs=fc8_MotionBlur,
                                     units=1,
                                     activation=None,
                                     kernel_initializer=tf.constant_initializer(
                                         np.transpose(state_dict["fc9_MotionBlur.weight"].numpy(), [1, 0])),
                                     bias_initializer=tf.constant_initializer(
                                         state_dict["fc9_MotionBlur.bias"].numpy())

                                     )

        # Object
        fc8_Object = tf.layers.dense(inputs=fc7,
                                     units=256,
                                     activation=tf.nn.relu,
                                     kernel_initializer=tf.constant_initializer(
                                         np.transpose(state_dict["fc8_Object.weight"].numpy(), [1, 0])),
                                     bias_initializer=tf.constant_initializer(state_dict["fc8_Object.bias"].numpy())
                                     )
        Object = tf.layers.dense(inputs=fc8_Object,
                                 units=1,
                                 activation=None,
                                 kernel_initializer=tf.constant_initializer(
                                     np.transpose(state_dict["fc9_Object.weight"].numpy(), [1, 0])),
                                 bias_initializer=tf.constant_initializer(
                                     state_dict["fc9_Object.bias"].numpy())

                                 )

        # Repetition
        fc8_Repetition = tf.layers.dense(inputs=fc7,
                                         units=256,
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.constant_initializer(
                                             np.transpose(state_dict["fc8_Repetition.weight"].numpy(), [1, 0])),
                                         bias_initializer=tf.constant_initializer(
                                             state_dict["fc8_Repetition.bias"].numpy())
                                         )
        Repetition = tf.layers.dense(inputs=fc8_Repetition,
                                     units=1,
                                     activation=None,
                                     kernel_initializer=tf.constant_initializer(
                                         np.transpose(state_dict["fc9_Repetition.weight"].numpy(), [1, 0])),
                                     bias_initializer=tf.constant_initializer(
                                         state_dict["fc9_Repetition.bias"].numpy())

                                     )
        # RuleOfThirds
        fc8_RuleOfThirds = tf.layers.dense(inputs=fc7,
                                           units=256,
                                           activation=tf.nn.relu,
                                           kernel_initializer=tf.constant_initializer(
                                               np.transpose(state_dict["fc8_RuleOfThirds.weight"].numpy(), [1, 0])),
                                           bias_initializer=tf.constant_initializer(
                                               state_dict["fc8_RuleOfThirds.bias"].numpy())
                                           )
        RuleOfThirds = tf.layers.dense(inputs=fc8_RuleOfThirds,
                                       units=1,
                                       activation=None,
                                       kernel_initializer=tf.constant_initializer(
                                           np.transpose(state_dict["fc9_RuleOfThirds.weight"].numpy(), [1, 0])),
                                       bias_initializer=tf.constant_initializer(
                                           state_dict["fc9_RuleOfThirds.bias"].numpy())

                                       )

        # Symmetry
        fc8_Symmetry = tf.layers.dense(inputs=fc7,
                                       units=256,
                                       activation=tf.nn.relu,
                                       kernel_initializer=tf.constant_initializer(
                                           np.transpose(state_dict["fc8_Symmetry.weight"].numpy(), [1, 0])),
                                       bias_initializer=tf.constant_initializer(state_dict["fc8_Symmetry.bias"].numpy())
                                       )
        Symmetry = tf.layers.dense(inputs=fc8_Symmetry,
                                   units=1,
                                   activation=None,
                                   kernel_initializer=tf.constant_initializer(
                                       np.transpose(state_dict["fc9_Symmetry.weight"].numpy(), [1, 0])),
                                   bias_initializer=tf.constant_initializer(
                                       state_dict["fc9_Symmetry.bias"].numpy())

                                   )

        # VividColor
        fc8_VividColor = tf.layers.dense(inputs=fc7,
                                         units=256,
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.constant_initializer(
                                             np.transpose(state_dict["fc8_VividColor.weight"].numpy(), [1, 0])),
                                         bias_initializer=tf.constant_initializer(
                                             state_dict["fc8_VividColor.bias"].numpy())
                                         )
        VividColor = tf.layers.dense(inputs=fc8_VividColor,
                                     units=1,
                                     activation=None,
                                     kernel_initializer=tf.constant_initializer(
                                         np.transpose(state_dict["fc9_VividColor.weight"].numpy(), [1, 0])),
                                     bias_initializer=tf.constant_initializer(
                                         state_dict["fc9_VividColor.bias"].numpy())

                                     )
        # Concatenating
        concat = tf.concat(
            [fc8, fc8_BalancingElement, fc8_ColorHarmony, fc8_Content, fc8_DoF, fc8_Light, fc8_MotionBlur, fc8_Object,
             fc8_Repetition, fc8_RuleOfThirds, fc8_Symmetry, fc8_VividColor], axis=1)

        f10_merge = tf.layers.dense(inputs=concat,
                                    units=128,
                                    activation=tf.nn.relu,
                                    kernel_initializer=tf.constant_initializer(
                                        np.transpose(state_dict["fc10_Merge.weight"].numpy(), [1, 0])),
                                    bias_initializer=tf.constant_initializer(
                                        state_dict["fc10_Merge.bias"].numpy())
                                    )
        fc11_score = tf.layers.dense(inputs=f10_merge,
                                     units=1,
                                     activation=None,
                                     kernel_initializer=tf.constant_initializer(
                                         np.transpose(state_dict["fc11_score.weight"].numpy(), [1, 0])),
                                     bias_initializer=tf.constant_initializer(
                                         state_dict["fc11_score.bias"].numpy())
                                     )
        # Collecting all output attributes
        output = {"Aesthetic": fc11_score,
                  "BalancingElement": BalancingElement,
                  "ColorHarmony": ColorHarmony,
                  "Content": Content,
                  "DoF": DoF,
                  "Light": Light,
                  "MotionBlur": MotionBlur,
                  "Object": Object,
                  "Repetition": Repetition,
                  "RuleOfThrids": RuleOfThirds,
                  "Symmetry": Symmetry,
                  "VividColor": VividColor}

        # Returning the requested one
        return output[attribute]


def aestheticsnet(image):
    model = AestheticsNet()
    return(model.aestheticsnet_fn(model.aestheticsnet_preprocess(image),"Aesthetic"))