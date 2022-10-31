"""Saliency Attack to refine perturbations in salient region"""
import argparse
import inspect
import json
import numpy as np
import os
from PIL import Image
import sys
import tensorflow as tf

import attacks
from tools.imagenet_labels import *
from tools.inception_v3_imagenet import model
from tools.utils import *

""" The collection of all attack classes """
ATTACK_CLASSES = [x for x in attacks.__dict__.values() if inspect.isclass(x)]
for attack in ATTACK_CLASSES:
  setattr(sys.modules[__name__], attack.__name__, attack)

""" Arguments """
IMAGENET_PATH = '/home/Dataset/ImageNet/'
NUM_LABELS = 1000
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

parser = argparse.ArgumentParser()

# Directory
parser.add_argument('--asset_dir', default='./assets', type=str)
parser.add_argument('--save_dir', default='./saves', type=str)
parser.add_argument('--save_noise_dir', default='./saves-noise', type=str)
parser.add_argument('--sal_dir', default='./saliency-maps', type=str)

# Experimental Setting
parser.add_argument('--img_index_start', default=0, type=int)
parser.add_argument('--sample_size', default=1000, type=int)
parser.add_argument('--save_img', dest='save_img', action='store_true')
parser.add_argument('--save_noise', dest='save_noise', action='store_true')

# Attack setting
parser.add_argument('--attack', default='SaliencyAttack', type=str, help='The type of attack')
parser.add_argument('--loss_func', default='cw', type=str, help='The type of loss function')
parser.add_argument('--epsilon', default=0.05, type=float, help='The maximum perturbation')
parser.add_argument('--max_queries', default=10000, type=int, help='The query limit')

# DFS Search Setting
parser.add_argument('--block_size', default=16, type=int, help='Initial block size')
parser.add_argument('--min_block_size', default=1, type=int, help='Initial minimal block size')

args = parser.parse_args()

if __name__ == '__main__':
  # Set verbosity
  tf.logging.set_verbosity(tf.logging.INFO)

  # Create a session
  sess = tf.InteractiveSession()

  # Build a graph
  x_input = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 3])
  y_input = tf.placeholder(dtype=tf.int32, shape=[None])

  logits, preds = model(sess, x_input)

  model = {
    'x_input': x_input,
    'y_input': y_input,
    'logits': logits,
    'preds': preds,
  }

  # Create attack
  attack_class = getattr(sys.modules[__name__], args.attack)
  attack = attack_class(model, args)

  # Create a directory
  if args.save_img:
    tf.gfile.MakeDirs(args.save_dir)
  if args.save_noise:
    tf.gfile.MakeDirs(args.save_noise_dir)

  # Print hyperparameters
  for key, val in vars(args).items():
    tf.logging.info('{}={}'.format(key, val))

  # Load the indices
  indices = np.load(os.path.join(args.asset_dir, 'indices_untargeted.npy'))

  # Main loop
  count = 0
  index = args.img_index_start
  total_num_corrects = 0
  total_queries = []
  total_l2 = []
  total_l0 = []
  index_to_query = {}
  count_small_saliency_maps = 0
  index_samll_saliency_maps = []

  while count < args.sample_size:
    tf.logging.info('')

    # Get an image and the corresponding label and saliency mask
    initial_img, orig_class = get_image(indices[index], IMAGENET_PATH)
    initial_img = np.expand_dims(initial_img, axis=0)
    sal_img = get_sal(indices[index], args.sal_dir)

    orig_class = np.expand_dims(orig_class, axis=0)

    count += 1

    # Run attack
    tf.logging.info('Untargeted attack on {}th image starts, index: {}, orig class: {}'.format(
      count, indices[index], label_to_name(orig_class[0])))
    adv_img, noise, num_queries, success, flag = attack.perturb(initial_img, sal_img, orig_class, indices[index], sess)

    # Record those saliency masks that have too small saliency region
    if flag:
      count_small_saliency_maps += 1
      index_samll_saliency_maps.append(indices[index])

    # Check if the adversarial example satisfies the constraint
    assert np.amax(np.abs(adv_img - initial_img)) <= args.epsilon + 1e-3
    assert np.amax(adv_img) <= 1. + 1e-3
    assert np.amin(adv_img) >= 0. - 1e-3
    p = sess.run(preds, feed_dict={x_input: adv_img})

    # Save the adversarial image
    if args.save_img:
      adv_image = Image.fromarray(np.ndarray.astype(adv_img[0, :, :, :] * 255, np.uint8))
      adv_image.save(os.path.join(args.save_dir, '{}_adv.jpg'.format(indices[index])))

    # Logging
    if success:
      total_num_corrects += 1
      total_queries.append(num_queries)
      l2 = np.sum(noise**2)**.5
      total_l2.append(l2)
      l0 = L0_dis(noise)
      total_l0.append(l0)
      index_to_query[indices[index]] = num_queries
      average_queries = 0 if len(total_queries) == 0 else np.mean(total_queries)
      median_queries = 0 if len(total_queries) == 0 else np.median(total_queries)
      average_l2 = 0 if len(total_l2) == 0 else np.mean(total_l2)
      average_l0 = 0 if len(total_l0) == 0 else np.mean(total_l0)
      success_rate = total_num_corrects / count
      print('Attack succeeds, final class: {}, avg queries: {:.4f}, med queries: {}, success rate: {:.4f}, '
                      'average L2: {:.4f}, average L0: {:.1f}%, No. of small_saliency_maps: {}, indexes: {}'.format(
        label_to_name(p[0]), average_queries, median_queries, success_rate, average_l2, average_l0*100, count_small_saliency_maps, index_samll_saliency_maps))

      # Save the successful noise
      if args.save_noise:
        noise = Image.fromarray(np.ndarray.astype((noise[0, :, :, :]) * 255, np.uint8))
        noise.save(os.path.join(args.save_noise_dir, '{}_noise.jpg'.format(indices[index])))

    else:
      index_to_query[indices[index]] = -1
      average_queries = 0 if len(total_queries) == 0 else np.mean(total_queries)
      median_queries = 0 if len(total_queries) == 0 else np.median(total_queries)
      average_l2 = 0 if len(total_l2) == 0 else np.mean(total_l2)
      average_l0 = 0 if len(total_l0) == 0 else np.mean(total_l0)
      success_rate = total_num_corrects / count
      print('Attack fails, final class: {}, avg queries: {:.4f}, med queries: {}, success rate: {:.4f}, '
                      'average L2: {:.4f}, average L0: {:.1f}%, No. of small_saliency_maps: {}, indexes: {}'.format(
        label_to_name(p[0]), average_queries, median_queries, success_rate, average_l2, average_l0*100, count_small_saliency_maps, index_samll_saliency_maps))

    index += 1