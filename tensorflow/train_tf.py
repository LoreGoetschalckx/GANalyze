import sys
import os
import argparse
import subprocess

import tensorflow as tf
import tensorflow_hub as hub

import numpy as np
import json
import datetime

from PIL import Image

sys.path.append('.')
import assessors
import generators
import transformations.tensorflow as transformations
import utils.tensorflow

# Collect command line arguments
#--------------------------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=str, default=0, help='which gpu to use.')
parser.add_argument('--num_samples', type=int, default="400000", help='number of samples to train for')
parser.add_argument('--checkpoint_resume', type=int, default=0, help='which checkpoint to load based on batch_start. -1 for latest stored checkpoint')
parser.add_argument('--train_alpha_a', type=float, default=-0.5, help='lower limit for step sizes to use during training')
parser.add_argument('--train_alpha_b', type=float, default=0.5, help='upper limit for step sizes to use during training')
parser.add_argument('--generator_arch', type=str, default="biggan", help='generator architecture')
parser.add_argument('--generator_model', type=str, default="biggan256", help='pretrained generator parameters to use, path or model_paths.json key')
parser.add_argument('--assessor', type=str, default="memnet", help='assessor function to compute the image property of interest')
parser.add_argument('--transformer', default=["OneDirection","None"], nargs=2,type=str, metavar=["name","arguments"], help="transformer function")

args = parser.parse_args()
opts = vars(args)

# Verify
if opts["checkpoint_resume"] != 0 and opts["checkpoint_resume"] != -1:
    assert(opts["checkpoint_resume"] % 4 ==0) # Needs to be a multiple of the batch size

# Choose GPU
os.environ["CUDA_VISIBLE_DEVICES"]=str(opts["gpu_id"])

# Creating directory to store checkpoints
with open("./generators/model_paths.json") as f:
    model_paths = json.load(f)
if opts["generator_model"] in model_paths.keys():
    model_sub_dir = opts["generator_model"] # already a shorthand for a model
else:
    model_sub_dir = os.path.basename(opts["generator_model"]) # making a shorthand

version = subprocess.check_output(["git", "describe","--always"]).strip().decode("utf-8")
checkpoint_dir = os.path.join("./checkpoints",opts["generator_arch"]+"__"+model_sub_dir,opts["assessor"],opts["transformer"][0]+"_"+opts["transformer"][1],version)

if opts["checkpoint_resume"]==0:
    os.makedirs(checkpoint_dir, exist_ok=False)

# Saving training settings
opts_file = os.path.join(checkpoint_dir, "opts.json")
opts["version"]=version
with open(opts_file, 'w') as fp:
    json.dump(opts, fp)

# Setting up file to store loss values
loss_file = os.path.join(checkpoint_dir,"losses.txt")

# Initialize tensorflow session
#-------------------------------------
tf.reset_default_graph()
sess = tf.InteractiveSession()

# Setting up Transformer
#--------------------------------------------------------------------------------------------------------------
transformer = opts["transformer"][0]
transformer_arguments = opts["transformer"][1]
if transformer_arguments != "None":
    key_value_pairs = transformer_arguments.split(",")
    key_value_pairs = [pair.split("=") for pair in key_value_pairs]
    transformer_arguments = {pair[0]:pair[1] for pair in key_value_pairs}
else:
    transformer_arguments={}

transformation = getattr(transformations,transformer)(**transformer_arguments)

# Setting up Generator
#--------------------------------------------------------------------------------------------------------------
generator = getattr(generators,opts["generator_arch"])(opts["generator_model"]) #why am I not doing train = false here?
categories = generator.get_categories()

# Setting up Assessor
#--------------------------------------------------------------------------------------------------------------
assessor = getattr(assessors,opts["assessor"])

# GANalyze model
#--------------------------------------------------------------------------------------------------------------
# inputs
inputs = generator.get_inputs()
z = inputs["z"]
y = inputs["y"]
step_sizes = tf.placeholder(tf.float32, shape=(None,1),name="step_sizes_ganalyze")
z_norm = tf.expand_dims(tf.norm(z, axis=1), axis=1)

# transformation
z_new = transformation.transform(z, z_norm, y, step_sizes)
z_new = tf.identity(z_new, name="z_new_ganalyze")
transformed_inputs = {k: v for k, v in inputs.items()} # deep copy
transformed_inputs["z"]=z_new

# gan part
    # original z
gan_image = generator.generate(inputs)
gan_image = tf.identity(gan_image, name="gan_image_ganalyze")
image_score = assessor(gan_image)
image_score = tf.identity(image_score, name="image_score_ganalyze")

    # transformed z
transformed_gan_image = generator.generate(transformed_inputs)
transformed_gan_image = tf.identity(transformed_gan_image, name="transformed_gan_image_ganalyze")
transformed_image_score = assessor(transformed_gan_image)
transformed_image_score = tf.identity(transformed_image_score, name="transformed_image_score_ganalyze")

# target assessor score
    # making sure all the tensors have the same number of dimensions
for i in range(len(image_score.get_shape().as_list())-2):
    if i ==0:
        step_sizes_expanded = tf.expand_dims(step_sizes,axis=-1)
    else:
        step_sizes_expanded =tf.expand_dims(step_sizes_expanded,axis=-1)
if len(image_score.get_shape().as_list()) ==2:
    step_sizes_expanded = step_sizes

step_sizes_expanded = tf.identity(step_sizes_expanded,name="expanded_step_sizes_ganalyze")
target = image_score + step_sizes_expanded
target = tf.identity(target, name="target_ganalyze")

# loss
loss = transformation.compute_loss(transformed_image_score, target)
loss = tf.identity(loss, name="loss_ganalyze")

# Training
#--------------------------------------------------------------------------------------------------------------
# optimizer
lr = 0.0002
train_step = tf.train.AdamOptimizer(lr).minimize(loss, var_list=tf.trainable_variables(scope="parameters_to_train"))
losses = utils.common.AverageMeter(name='Loss')

# initialize
utils.tensorflow.initialize_uninitialized(sess) # why are some of the biggan tensor not initialized?

# set up saver
saver = tf.train.Saver()

# figure out where to resume
if opts["checkpoint_resume"]==0:
    checkpoint_resume = 0
elif opts["checkpoint_resume"]==-1:
    saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))
    latest_checkpoint_number = int(str.split(tf.train.latest_checkpoint(checkpoint_dir),"-")[-1])
    checkpoint_resume=latest_checkpoint_number
else:
    saver.restore(sess, os.path.join(checkpoint_dir,"model.ckpt-"+str(opts["checkpoint_resume"])))
    checkpoint_resume=opts["checkpoint_resume"]

# training settings
optim_iter = 0
batch_size = 4
train_alpha_a = opts["train_alpha_a"]
train_alpha_b = opts["train_alpha_b"]
num_samples = opts["num_samples"]

# create training set
training_set = generator.create_training_set(num_samples)

# iterate over batches
for batch_start in range(0,num_samples,batch_size):

    # skip batches we have already done (when resuming)
    if batch_start <= checkpoint_resume and checkpoint_resume != 0:
        optim_iter = optim_iter + 1
        continue

    # create feed_dict
    step_sizes_np = (train_alpha_b - train_alpha_a) * np.random.random(size=(batch_size, 1)) + train_alpha_a
    s = slice(batch_start, min(num_samples, batch_start + batch_size))
    feed_dict = {k: (v[s] if type(v) is np.ndarray else v) for k,v in training_set.items()}
    feed_dict[step_sizes] = step_sizes_np

    # evaluate
    curr_loss, _ = sess.run([loss, train_step], feed_dict=feed_dict)

    # print loss
    losses.update(curr_loss, batch_size)
    print(f'[{batch_start}/{num_samples}] {losses}')

    # saving
    transformation.loss_to_file(feed_dict=feed_dict,batch_start=batch_start,lossfile=loss_file,sess=sess)
    if optim_iter % 25000 == 0:
        print("saving checkpoint")
        image = sess.run(gan_image,feed_dict=feed_dict)
        image = np.uint8(image)
        image = Image.fromarray(image[0,:])
        image.save(os.path.join(checkpoint_dir,"example"+str(batch_start)+".jpg"))
        saver.save(sess, os.path.join(checkpoint_dir, "model.ckpt"), global_step=batch_start)
    optim_iter = optim_iter + 1

saver.save(sess, os.path.join(checkpoint_dir, "model.ckpt"), global_step=num_samples)
















