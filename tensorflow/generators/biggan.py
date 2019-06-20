import sys
import os
import numpy as np
import json

import tensorflow as tf
import tensorflow_hub as hub

sys.path.append('..')
import utils.tensorflow
import utils.common


__all__ = ['BigGan', 'biggan']

class BigGan():
    def __init__(self, path, train):

        if train:
            model_paths_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "model_paths.json")
            with open(model_paths_file) as f:
                model_paths = json.load(f)

            if path in model_paths.keys():
                self.model_path = model_paths[path]
            else:
                self.model_path = path
            self.module = hub.Module(self.model_path)
            self.inputs = self.gather_inputs_train()
        else:
            self.inputs = self.gather_inputs_test()

    def get_inputs(self):
        return self.inputs

    def gather_inputs_train(self):
        # TO DO: how to make this private?
        inputs = {k: tf.placeholder(v.dtype, v.get_shape().as_list(), k)
                  for k, v in self.module.get_input_info_dict().items()}

        z = tf.identity(inputs["z"],name="z_ganalyze")
        truncation = tf.identity(inputs["truncation"],name="truncation_ganalyze")
        y = tf.identity(inputs["y"],name="y_ganalyze")

        return {"z": z, "truncation": truncation, "y": y}

    def gather_inputs_test(self):
        z = tf.get_default_graph().get_tensor_by_name("z_ganalyze:0")
        truncation = tf.get_default_graph().get_tensor_by_name("truncation_ganalyze:0")
        y = tf.get_default_graph().get_tensor_by_name("y_ganalyze:0")
        return{"z":z, "truncation":truncation, "y":y}

    def get_categories(self):
        categories_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "categories_imagenet.txt")
        categories = [x.strip() for x in open(categories_file)]
        return categories

    def denormalize(self,biggan_output):
        return tf.clip_by_value(((biggan_output + 1) / 2.0) * 256, 0, 255)

    def generate(self,inputs):
        return self.denormalize(self.module(inputs))

    def create_training_set(self,num_samples):
        np.random.seed(seed=0)
        truncation = 1.0
        vocab_size = self.inputs["y"].get_shape().as_list()[1]
        dim_z = self.inputs["z"].get_shape().as_list()[1]
        zs = utils.common.truncated_z_sample(num_samples,dim_z, truncation)
        ys = np.random.randint(0, vocab_size, size=zs.shape[0])
        ys = utils.tensorflow.one_hot_if_needed(ys, vocab_size)

        training_set = {}
        training_set[self.inputs["z"]]=zs
        training_set[self.inputs["y"]] = ys
        training_set[self.inputs["truncation"]] = truncation

        return training_set

    def create_feeddict(self,num_samples,y,truncation):
        vocab_size = self.inputs["y"].get_shape().as_list()[1]
        dim_z = self.inputs["z"].get_shape().as_list()[1]
        zs = utils.common.truncated_z_sample(num_samples, dim_z, truncation)
        ys = np.repeat(y, num_samples)
        ys = utils.tensorflow.one_hot_if_needed(ys, vocab_size)
        feeddicts = []
        for batch_start in range(0,num_samples,4):
            s = slice(batch_start, min(num_samples, batch_start + 4))
            feeddicts.append({self.inputs["z"]:zs[s], self.inputs["y"]:ys[s], self.inputs["truncation"]:truncation})
        return(feeddicts)

def biggan(path,train=True):
    instance = BigGan(path, train)
    return instance
