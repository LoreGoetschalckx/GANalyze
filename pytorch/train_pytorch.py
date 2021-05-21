import argparse
import json
import os
import subprocess

import numpy as np
import torch
import torch.optim as optim

import assessors
import generators
import transformations.pytorch as transformations
import utils.common
import utils.pytorch

# Collect command line arguments
# --------------------------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=str, default=0, help='which gpu to use.')
parser.add_argument('--num_samples', type=int, default="400000", help='number of samples to train for')
parser.add_argument('--checkpoint_resume', type=int, default=0, help='which checkpoint to load based on batch_start. -1 for latest stored checkpoint')
parser.add_argument('--train_alpha_a', type=float, default=-0.5, help='lower limit for step sizes to use during training')
parser.add_argument('--train_alpha_b', type=float, default=0.5, help='upper limit for step sizes to use during training')
parser.add_argument('--generator', default=["biggan256", "None"], nargs=2, type=str, metavar=["name", "arguments"], help='generator function to use')
parser.add_argument('--assessor', type=str, default="emonet", help='assessor function to compute the image property of interest')
parser.add_argument('--transformer', default=["OneDirection", "None"], nargs=2, type=str, metavar=["name", "arguments"], help="transformer function")

args = parser.parse_args()
opts = vars(args)

# Verify
if opts["checkpoint_resume"] != 0 and opts["checkpoint_resume"] != -1:
    assert(opts["checkpoint_resume"] % 4 == 0)  # Needs to be a multiple of the batch size

# Choose GPU
if opts["gpu_id"] != -1:
    device = torch.device("cuda:" + str(opts["gpu_id"]) if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")

# Creating directory to store checkpoints
version = subprocess.check_output(["git", "describe", "--always"]).strip().decode("utf-8")
checkpoint_dir = os.path.join(
    "./checkpoints",
    "_".join(opts["generator"]),
    opts["assessor"],
    "_".join(opts["transformer"]),
    version)

if opts["checkpoint_resume"] == 0:
    os.makedirs(checkpoint_dir, exist_ok=False)

# Saving training settings
opts_file = os.path.join(checkpoint_dir, "opts.json")
opts["version"] = version
with open(opts_file, 'w') as fp:
    json.dump(opts, fp)

# Setting up file to store loss values
loss_file = os.path.join(checkpoint_dir, "losses.txt")

# Some characteristics
# --------------------------------------------------------------------------------------------------------------
dim_z = {
    'biggan256': 140,
    'biggan512': 128
}.get(opts['generator'][0])

vocab_size = {'biggan256': 1000, 'biggan512': 1000}.get(opts['generator'][0])

# Setting up Transformer
# --------------------------------------------------------------------------------------------------------------
transformer = opts["transformer"][0]
transformer_arguments = opts["transformer"][1]
if transformer_arguments != "None":
    key_value_pairs = transformer_arguments.split(",")
    key_value_pairs = [pair.split("=") for pair in key_value_pairs]
    transformer_arguments = {pair[0]: pair[1] for pair in key_value_pairs}
else:
    transformer_arguments = {}

transformation = getattr(transformations, transformer)(dim_z, vocab_size, **transformer_arguments)
transformation = transformation.to(device)

# Setting up Generator
# --------------------------------------------------------------------------------------------------------------
generator = opts["generator"][0]
generator_arguments = opts["generator"][1]
if generator_arguments != "None":
    key_value_pairs = generator_arguments.split(",")
    key_value_pairs = [pair.split("=") for pair in key_value_pairs]
    generator_arguments = {pair[0]: pair[1] for pair in key_value_pairs}
else:
    generator_arguments = {}

generator = getattr(generators, generator)(**generator_arguments)

for p in generator.parameters():
    p.requires_grad = False
generator.eval()
generator = generator.to(device)

# Setting up Assessor
# --------------------------------------------------------------------------------------------------------------
assessor_elements = getattr(assessors, opts['assessor'])(True)
if isinstance(assessor_elements, tuple):
    assessor = assessor_elements[0]
    input_transform = assessor_elements[1]
    output_transform = assessor_elements[2]
else:
    assessor = assessor_elements

    def input_transform(x):
        return x  # identity, no preprocessing

    def output_transform(x):
        return x  # identity, no postprocessing

if hasattr(assessor, 'parameters'):
    for p in assessor.parameters():
        p.requires_grad = False
        assessor.eval()
        assessor.to(device)

# Training
# --------------------------------------------------------------------------------------------------------------
# optimizer
optimizer = optim.Adam(transformation.parameters(), lr=0.0002)
losses = utils.common.AverageMeter(name='Loss')

# figure out where to resume
if opts["checkpoint_resume"] == 0:
    checkpoint_resume = 0
elif opts["checkpoint_resume"] == -1:
    available_checkpoints = [x for x in os.listdir(checkpoint_dir) if x.endswith(".pth")]
    available_batch_numbers = [x.split('.')[0].split("_")[-1] for x in available_checkpoints]
    latest_number = max(available_batch_numbers)
    file_to_load = available_checkpoints[available_batch_numbers.index(latest_number)]
    transformation.load_state_dict(torch.load(os.path.join(checkpoint_dir, file_to_load)))
    checkpoint_resume = latest_number
else:
    transformation.load_state_dict(torch.load(os.path.join(checkpoint_dir,
                                                           "pytorch_model_{}.pth".format(opts["checkpoint_resume"]))))
    checkpoint_resume = opts["checkpoint_resume"]

#  training settings
optim_iter = 0
batch_size = 4
train_alpha_a = opts["train_alpha_a"]
train_alpha_b = opts["train_alpha_b"]
num_samples = opts["num_samples"]

# create training set
np.random.seed(seed=0)
truncation = 1
zs = utils.common.truncated_z_sample(num_samples, dim_z, truncation)
ys = np.random.randint(0, vocab_size, size=zs.shape[0])

# loop over data batches
for batch_start in range(0, num_samples, batch_size):

    # zero the parameter gradients
    optimizer.zero_grad()

    # skip batches we've already done (this would happen when resuming from a checkpoint)
    if batch_start <= checkpoint_resume and checkpoint_resume != 0:
        optim_iter = optim_iter + 1
        continue

    # input batch
    s = slice(batch_start, min(num_samples, batch_start + batch_size))
    z = torch.from_numpy(zs[s]).type(torch.FloatTensor).to(device)
    y = torch.from_numpy(ys[s]).to(device)
    step_sizes = (train_alpha_b - train_alpha_a) * \
        np.random.random(size=(batch_size)) + train_alpha_a  # sample step_sizes
    step_sizes_broadcast = np.repeat(step_sizes, dim_z).reshape([batch_size, dim_z])
    step_sizes_broadcast = torch.from_numpy(step_sizes_broadcast).type(torch.FloatTensor).to(device)

    # ganalyze steps
    gan_images = generator(z, utils.pytorch.one_hot(y))
    gan_images = input_transform(utils.pytorch.denorm(gan_images))
    gan_images = gan_images.view(-1, *gan_images.shape[-3:])
    gan_images = gan_images.to(device)
    out_scores = output_transform(assessor(gan_images)).to(device).float()
    target_scores = out_scores + torch.from_numpy(step_sizes).to(device).float()

    z_transformed = transformation.transform(z, utils.pytorch.one_hot(y), step_sizes_broadcast)
    gan_images_transformed = generator(z_transformed, utils.pytorch.one_hot(y))
    gan_images_transformed = input_transform(utils.pytorch.denorm(gan_images_transformed))
    gan_images_transformed = gan_images_transformed.view(-1, *gan_images_transformed.shape[-3:])
    gan_images_transformed = gan_images_transformed.to(device)
    out_scores_transformed = output_transform(assessor(gan_images_transformed)).to(device).float()

    # compute loss
    loss = transformation.compute_loss(out_scores_transformed, target_scores, batch_start, loss_file)

    # backwards
    loss.backward()
    optimizer.step()

    # print loss
    losses.update(loss.item(), batch_size)
    print(f'[{batch_start}/{num_samples}] {losses}')

    if optim_iter % 250 == 0:
        print("saving checkpoint")
        torch.save(transformation.state_dict(), os.path.join(checkpoint_dir, "pytorch_model_{}.pth".format(batch_start)))
    optim_iter = optim_iter + 1

torch.save(transformation.state_dict(), os.path.join(checkpoint_dir, "pytorch_model_{}.pth".format(opts["num_samples"])))
