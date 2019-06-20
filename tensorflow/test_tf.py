import sys
import os
import argparse
import subprocess

import tensorflow as tf
import tensorflow_hub as hub

import numpy as np
import json
import ast

import PIL.ImageDraw
import PIL.ImageFont

sys.path.append('.')
import generators
import utils.tensorflow
import utils.common
import transformations.tensorflow as transformations

# Collect command line arguments
#--------------------------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=str, default=0, help='which gpu to use.')
parser.add_argument('--alpha', type=float, default=0.1, help='stepsize for testing')
parser.add_argument('--test_truncation', type=float, default=1, help='truncation to use in test phase')
parser.add_argument('--checkpoint_dir', type=str, default="", help='path for directory with the checkpoints of the trained model we want to use')
parser.add_argument('--checkpoint', type=int, default=400000, help='which checkpoint to load')
parser.add_argument('--mode', default="bigger_step", choices=["iterative","bigger_step"], help="how to make the test sequences. bigger_step was used in the paper.")

args = parser.parse_args()
opts = vars(args)
print(opts)

# Choose GPU
os.environ["CUDA_VISIBLE_DEVICES"]=str(opts["gpu_id"])

# Creating directory to store output visualizations
train_opts_file = os.path.join(opts["checkpoint_dir"], "opts.json")
with open(train_opts_file) as f:
    train_opts = json.load(f)

if not isinstance(train_opts["transformer"],list):
    train_opts["transformer"]=[train_opts["transformer"]]

with open("./generators/model_paths.json") as f:
    model_paths = json.load(f)

if train_opts["generator_model"] in model_paths.keys():
    model_sub_dir = train_opts["generator_model"] # already a shorthand for a model
else:
    model_sub_dir = os.path.basename(train_opts["generator_model"]) # making a shorthand
test_version = str(subprocess.check_output(["git", "describe","--always"]).strip())
result_dir = os.path.join("./output",train_opts["generator_arch"]+"__"+model_sub_dir,train_opts["assessor"],"-".join(train_opts["transformer"]),train_opts["version"],"alpha_"+str(opts["alpha"])+"_truncation_"+str(opts["test_truncation"])+"_iteration_"+str(opts["checkpoint"])+"_"+"_"+opts["mode"])

os.makedirs(result_dir, exist_ok=False)

# Saving testing settings
opts_file = os.path.join(result_dir, "opts.json")
opts["test_version"]=test_version
with open(opts_file, 'w') as fp:
    json.dump(opts, fp)

# Initialize tensorflow session and restoring GANalyze graph
#--------------------------------------------------------------------------------------------------------------
tf.reset_default_graph()
sess = tf.InteractiveSession()
new_saver = tf.train.import_meta_graph(os.path.join(opts["checkpoint_dir"],"model.ckpt-"+str(opts["checkpoint"])+'.meta'))
new_saver.restore(sess,os.path.join(opts["checkpoint_dir"],"model.ckpt-"+str(opts["checkpoint"])))

# Setting up generator
#--------------------------------------------------------------------------------------------------------------
train_opts_file = os.path.join(opts["checkpoint_dir"], "opts.json") # check which one was used for training
with open(train_opts_file) as f:
    train_opts = json.load(f)
generator = getattr(generators, train_opts["generator_arch"])(train_opts["generator_model"],train=False)
categories = generator.get_categories()

# some characteristics
inputs = generator.get_inputs()
dim_z = inputs["z"].get_shape().as_list()[-1]
vocab_size = inputs["y"].get_shape().as_list()[-1]
categories = generator.get_categories()

# Graph nodes of interest
#--------------------------------------------------------------------------------------------------------------
# nodes of interest
input_step_sizes = tf.get_default_graph().get_tensor_by_name("step_sizes_ganalyze:0")
gan_image = tf.get_default_graph().get_tensor_by_name("gan_image_ganalyze:0")
image_score = tf.get_default_graph().get_tensor_by_name("image_score_ganalyze:0")
z_new = tf.get_default_graph().get_tensor_by_name("z_new_ganalyze:0")
transformed_gan_image = tf.get_default_graph().get_tensor_by_name("transformed_gan_image_ganalyze:0")
transformed_image_score = tf.get_default_graph().get_tensor_by_name("transformed_image_score_ganalyze:0")

# Testing
#--------------------------------------------------------------------------------------------------------------
num_samples = 10
truncation = opts["test_truncation"]
iters = 3 #number of clone images to generate (both positive and negative)
np.random.seed(seed=999)
step_sizes = np.array([[opts["alpha"]]])

if vocab_size == 0:
    num_categories = 1
else:
    num_categories = vocab_size

for y in range(num_categories):

    # gather inputs
    feed_dicts = generator.create_feeddict(num_samples,y,truncation)

    ims = []
    outscores = []

    for feed_dict in feed_dicts:
        ims_batch = []
        outscores_batch = []
        z_start = feed_dict[inputs["z"]]
        feed_dict[input_step_sizes] = step_sizes

        if opts["mode"]=="iterative":
            print("iterative")

            # original seed image
            x, tmp, outscore = sess.run([gan_image, z_new, image_score], feed_dict=feed_dict)
            x = np.uint8(x)
            outscore = np.expand_dims(np.mean(outscore,tuple(list(range(1,len(outscore.shape))))),1)
            ims_batch.append(utils.common.annotate_outscore(x, outscore))
            outscores_batch.append(outscore)

            # negative clone images
            z_next = z_start
            step_sizes = -step_sizes
            for iter in range(0, iters, 1):
                feed_dict[input_step_sizes]=step_sizes
                feed_dict[inputs["z"]] = z_next
                x, tmp, outscore = sess.run([transformed_gan_image, z_new, transformed_image_score], feed_dict=feed_dict)
                x = np.uint8(x)
                outscore =  np.expand_dims(np.mean(outscore, tuple(list(range(1, len(outscore.shape))))), 1)
                z_next = tmp
                ims_batch.append(utils.common.annotate_outscore(x, outscore))
                outscores_batch.append(outscore)

            ims_batch.reverse()

            # positive clone images
            step_sizes = -step_sizes
            z_next = z_start
            for iter in range(0, iters, 1):
                feed_dict[input_step_sizes]=step_sizes
                feed_dict[inputs["z"]] = z_next
                x, tmp, outscore = sess.run([transformed_gan_image, z_new, transformed_image_score], feed_dict=feed_dict)
                x = np.uint8(x)
                outscore =  np.expand_dims(np.mean(outscore, tuple(list(range(1, len(outscore.shape))))), 1)
                z_next = tmp
                ims_batch.append(utils.common.annotate_outscore(x, outscore))
                outscores_batch.append(outscore)

        else:
            print("bigger_step")

            # original seed image
            x, outscore = sess.run([gan_image, image_score], feed_dict=feed_dict)
            x = np.uint8(x)
            outscore = np.expand_dims(np.mean(outscore, tuple(list(range(1, len(outscore.shape))))), 1)
            ims_batch.append(utils.common.annotate_outscore(x, outscore))
            outscores_batch.append(outscore)

            # negative clone images
            step_sizes = -step_sizes
            for iter in range(0, iters, 1):
                feed_dict[input_step_sizes] = step_sizes*(iter+1)
                x, outscore = sess.run([transformed_gan_image, transformed_image_score], feed_dict=feed_dict)
                x = np.uint8(x)
                outscore =  np.expand_dims(np.mean(outscore, tuple(list(range(1, len(outscore.shape))))), 1)
                ims_batch.append(utils.common.annotate_outscore(x, outscore))
                outscores_batch.append(outscore)

            ims_batch.reverse()

            # positive clone images
            step_sizes = -step_sizes
            for iter in range(0, iters, 1):
                feed_dict[input_step_sizes] = step_sizes*(iter+1)
                x, outscore = sess.run([transformed_gan_image, transformed_image_score], feed_dict=feed_dict)
                x = np.uint8(x)
                outscore =  np.expand_dims(np.mean(outscore, tuple(list(range(1, len(outscore.shape))))), 1)
                ims_batch.append(utils.common.annotate_outscore(x, outscore))
                outscores_batch.append(outscore)


        ims_batch = [np.expand_dims(im, 0) for im in ims_batch]
        ims_batch = np.concatenate(ims_batch, axis=0)
        ims_batch = np.transpose(ims_batch, (1, 0, 2, 3, 4))
        ims.append(ims_batch)

    ims = np.concatenate(ims, axis=0)

    ims_final = ims
    ims_final = np.reshape(ims_final, (ims.shape[0] * ims.shape[1], ims.shape[2], ims.shape[3], ims.shape[4]))
    os.makedirs(result_dir, exist_ok=True)
    I = PIL.Image.fromarray(utils.common.imgrid(ims_final, cols=iters * 2 + 1))
    I.save(os.path.join(result_dir, categories[y] + ".jpg"))
    print("y: ", y)

