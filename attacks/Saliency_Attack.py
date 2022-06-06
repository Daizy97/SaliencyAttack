import cv2
import itertools
import heapq
import math
import sys
import numpy as np
import tensorflow as tf

from attacks.SaliencyAttack_helper import Node

class SaliencyAttack(object):
  """
  Saliency Attack to refine perturbations in salient region
  Note that since heapq library only supports min heap, we flip the sign of loss function.
  """

  def __init__(self, model, args, **kwargs):
    """Initialize attack.

    Args:
      model: TensorFlow model
      args: arguments
    """
    # Hyperparameter setting
    self.min_block_size = args.min_block_size
    self.loss_func = args.loss_func
    self.max_queries = args.max_queries
    self.epsilon = args.epsilon
    self.block_size = args.block_size

    # Create helper
    self.node = Node(model, args)

    # Network setting
    self.x_input = model['x_input']
    self.y_input = model['y_input']
    self.logits = model['logits']
    self.preds = model['preds']

    probs = tf.nn.softmax(self.logits)
    batch_num = tf.range(0, limit=tf.shape(probs)[0])
    indices = tf.stack([batch_num, self.y_input], axis=1)
    ground_truth_probs = tf.gather_nd(params=probs, indices=indices)
    top_2 = tf.nn.top_k(probs, k=2)
    max_indices = tf.where(tf.equal(top_2.indices[:, 0], self.y_input), top_2.indices[:, 1], top_2.indices[:, 0])
    max_indices = tf.stack([batch_num, max_indices], axis=1)
    max_probs = tf.gather_nd(params=probs, indices=max_indices)

    # loss function
    if self.loss_func == 'xent':
      self.losses = -tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=self.logits, labels=self.y_input)
    elif self.loss_func == 'cw':
      self.losses = tf.log(ground_truth_probs+1e-10) - tf.log(max_probs+1e-10)
    else:
      tf.logging.info('Loss function must be xent or cw')
      sys.exit()

  def _perturb_image(self, image, noise):
    """Given an image and a noise, generate a perturbed image.
    First, resize the noise with the size of the image.
    Then, add the resized noise to the image.

    Args:
      image: numpy array of size [1, 299, 299, 3], an original image
      noise: numpy array of size [1, 256, 256, 3], a noise

    Returns:
      adv_iamge: numpy array of size [1, 299, 299, 3], an perturbed image
    """
    adv_image = image + cv2.resize(noise[0, ...], (self.width, self.height), interpolation=cv2.INTER_NEAREST)
    adv_image = np.clip(adv_image, 0., 1.)
    return adv_image

  def _split_block(self, upper_left, lower_right, block_size, sal_img):
    """Split an image into a set of blocks.
    Note that a block consists of [upper_left, lower_right, channel]

    Args:
      upper_left: [x, y], the coordinate of the upper left of an image
      lower_right: [x, y], the coordinate of the lower right of an image
      block_size: int, the size of a block
      sal_img: [256, 256], saliency mask, {0,1} for each pixels

    Returns:
      blocks: list, the set of blocks
    """
    blocks = []
    xs = np.arange(upper_left[0], lower_right[0], block_size)
    ys = np.arange(upper_left[1], lower_right[1], block_size)
    for x, y in itertools.product(xs, ys):

      block = np.ones([block_size, block_size], dtype='int32')
      block = block * sal_img[x:x + block_size, y:y + block_size]

      for c in range(3):
        blocks.append([[x, y], [x + block_size, y + block_size], c, block])
    return blocks

  def _filter_block(self, blocks_all):
    """Choose the blocks in saliency region.

    Args:
      blocks_all: list, the set of blocks containing saliency mask
      Note that a block consists of [upper_left, lower_right, channel, saliency_mask]

    Returns:
      blocks: list, the set of blocks
    """
    blocks = []
    for i, block in enumerate(blocks_all):
      _, _, _, saliency_block = block
      if np.sum(saliency_block == 1) > 0:  # at least one pixel in saliency region
        blocks.append(block)

    return blocks

  def _filter_finer_block(self, blocks_all, coarse_block):
    """choose finer blocks in the coarse_block region

    Args:
      blocks_all: list, the set of all finer blocks containing saliency mask
      coarse_block: [upper_left, lower_right, channel, saliency_mask], one coarse block
      Note that a block consists of [upper_left, lower_right, channel, saliency_mask]

    Returns:
      blocks: list, the set of chosen finer blocks
    """
    blocks = []
    upper_left, lower_right, channel, _ = coarse_block
    for i, block in enumerate(blocks_all):
      ul, lr, c, _ = block
      if c == channel and (ul[0] >= upper_left[0] and lr[0] <= lower_right[0]) and (ul[1] >= upper_left[1] and lr[1] <= lower_right[1]):
        blocks.append(block)

    return blocks

  def _add_epsilon_noise(self, noise, block, flag):
    """Flip the sign of perturbation on a block.
    Args:
      noise: numpy array of size [1, 256, 256, 3], a noise
      block: [upper_left, lower_right, channel, saliency_mask], a block
      flag: int, {1, -1}, add positive or negative epsilon

    Returns:
      noise: numpy array of size [1, 256, 256, 3], an updated noise
    """
    noise_new = np.copy(noise)
    upper_left, lower_right, channel, saliency_block = block
    noise_new[0, upper_left[0]:lower_right[0], upper_left[1]:lower_right[1], channel] = flag * self.epsilon * saliency_block
    return noise_new

  def _flip_noise(self, noise, block):
    """Flip the sign of perturbation on a block.
    Args:
      noise: numpy array of size [1, 256, 256, 3], a noise
      block: [upper_left, lower_right, channel, saliency_mask], a block

    Returns:
      noise: numpy array of size [1, 256, 256, 3], an updated noise
    """
    noise_new = np.copy(noise)
    upper_left, lower_right, channel, _ = block
    noise_new[0, upper_left[0]:lower_right[0], upper_left[1]:lower_right[1], channel] *= -1
    return noise_new

  def perturb(self, image, sal_img, label, index, sess):
    """Perturb an image.

    Args:
      image: numpy array of size [1, 299, 299, 3], an original image
      sal_img: saliency mask of size [256, 256]
      label: numpy array of size [1], the label of the image (or target label)
      index: int, the index of the image
      sess: TensorFlow session

    Returns:
      adv_image: numpy array of size [1, 299, 299, 3], an adversarial image
      num_queries: int, the number of queries
      success: bool, True if attack is successful
      flag: boo, Ture if the salient region is too small
    """
    # Set random seed by index for the reproducibility
    np.random.seed(index)

    # Class variables
    self.width = image.shape[1]
    self.height = image.shape[2]

    # Local variables
    adv_image = np.copy(image)
    num_queries = 0
    block_size = self.block_size
    min_block_size = self.min_block_size
    upper_left = [0, 0]
    lower_right = [256, 256]

    # Initialize a noise with zeros
    noise = np.zeros([1, 256, 256, 3], dtype=np.float32)

    # Preprocess for saliency mask
    flag = 0  # record whether the saliency region is too small
    sal_img = np.where(sal_img == 255, 1, 0)
    num_saliency = np.sum(sal_img == 1)
    num_pixels = 256 * 256
    # if the saliency region is too small in the image, we just perturb the whole image region
    if num_saliency / num_pixels <= 0.01:
      sal_img = np.ones([256, 256], dtype='int32')
      flag = 1

    # # if perturb in the whole image:
    # sal_img = np.ones([256, 256], dtype='int32')

    # Initialize a 2D empty list
    # e.g assume block_size=64, min_block_size=2,
    # then we at most have log2(64)-log2(2)+1 hierarchies
    maximal_num_hierarchies = int(math.log(block_size, 2) - math.log(min_block_size, 2) + 1)
    blocks = [[] for i in range(maximal_num_hierarchies)]
    num_blocks = [0 for i in range(maximal_num_hierarchies)]

    # Split an image into a set of blocks at all hierarchies
    for i in range(maximal_num_hierarchies):
      blocks_all = self._split_block(upper_left, lower_right, block_size, sal_img)
      blocks[i] = self._filter_block(blocks_all)
      num_blocks[i] = len(blocks[i])
      block_size //= 2

    # Refining on initial blocks
    num_initial_blocks = num_blocks[0] * 2
    image_batch = np.zeros([num_initial_blocks, self.width, self.height, 3], np.float32)
    noise_batch = np.zeros([num_initial_blocks, 256, 256, 3], np.float32)
    label_batch = np.tile(label, num_initial_blocks)
    for i in range(0, num_initial_blocks, 2):
      # add +epsilon
      noise_batch[i] = self._add_epsilon_noise(noise, blocks[0][i//2], 1)
      image_batch[i] = self._perturb_image(image, noise_batch[i:i+1])
      # add -epsilon
      noise_batch[i+1] = self._add_epsilon_noise(noise, blocks[0][i//2], -1)
      image_batch[i+1] = self._perturb_image(image, noise_batch[i+1:i+2])

    if num_initial_blocks > 100:
      losses = np.zeros(num_initial_blocks, dtype=np.float32)
      preds = np.zeros(num_initial_blocks, dtype=np.int64)
      batch_size = 100
      num_batches = int(math.ceil(num_initial_blocks / batch_size))
      for ibatch in range(num_batches):
        bstart = ibatch * batch_size
        if (num_initial_blocks-bstart) < batch_size:
          batch_size = num_initial_blocks - bstart
        bend = bstart + batch_size
        losses[bstart:bend], preds[bstart:bend] = sess.run([self.losses, self.preds],
          feed_dict={self.x_input: image_batch[bstart:bend], self.y_input: label_batch[bstart:bend]})
    else:
      losses, preds = sess.run([self.losses, self.preds],
        feed_dict={self.x_input: image_batch, self.y_input: label_batch})

    # Early stopping
    success_indices,  = np.where(preds != label)
    if len(success_indices) > 0:
      noise[0, ...] = noise_batch[success_indices[0], ...]
      curr_loss = losses[success_indices[0]]
      num_queries += success_indices[0] + 1
      adv_image = self._perturb_image(image, noise)
      tf.logging.info("num queries: {}".format(num_queries))
      return adv_image, noise, num_queries, True, flag

    num_queries += num_initial_blocks

    # Initialize a 2D empty list
    # priority_queue[0]: record (losses[idx], idx) of initial blocks
    # priority_queue[i+1] (i>0): record (margin[idx], idx) of the finer blocks extracted from the best block in priority_queue[i];
    # margin = losses[idx] - best_loss
    priority_queue = [[] for i in range(maximal_num_hierarchies)]

    # Push into the priority queue
    for i in range(num_initial_blocks):
      heapq.heappush(priority_queue[0], (losses[i], i))
    rec_queue = priority_queue[0].copy()  # record the initial blocks for later runs

    # Refining on further blocks
    curr_loss, idx = heapq.heappop(priority_queue[0])
    best_loss = curr_loss
    curr_noise = noise_batch[idx:idx+1]
    best_noise = curr_noise
    curr_block = blocks[0][idx//2]

    # loop start: iterate on those initial blocks that have been perturbed in last run
    i_run = 1
    skip = 0  # No. of hierarchies to be skipped
    maximal_skip_num = int(math.log(self.block_size, 2) - 1)  # at most skip to block_size = 2^1
    while num_queries <= self.max_queries:
      tf.logging.info("----------------------------------------------------------------")
      tf.logging.info("{}th run with block_size {}:".format(i_run, self.block_size/pow(2, skip)))

      ## Refining on first hierarchy
      for i in range(num_initial_blocks):
        # clear the previous priority_queue except for the 1st hierarchy
        for id in range(1, maximal_num_hierarchies):
          priority_queue[id].clear()
        if len(priority_queue[0]) == 0:
          break
        if i_run ==1 and i > 0:  # skip the first best block
          tf.logging.info("After {} initial blocks, num queries: {}, loss: {:.4f}".format(i, num_queries, curr_loss))
          _, idx = heapq.heappop(priority_queue[0])
          sign = [1, -1]
          curr_block = blocks[0][idx//2]
          curr_noise = self._add_epsilon_noise(curr_noise, curr_block, sign[idx%2])

          image_test = self._perturb_image(image, curr_noise)
          losses, pred = sess.run([self.losses, self.preds],
            feed_dict={self.x_input: image_test, self.y_input: label})
          num_queries += 1
          # Early stopping
          s = (pred != label)
          if s:
            tf.logging.info("num queries: {}".format(num_queries))
            return image_test, curr_noise, num_queries, True, flag

          # tf.logging.info("After {} initial blocks, num queries: {}, loss: {:.4f}".format(i, num_queries, curr_loss))
          # clear the previous priority_queue except for the 1st hierarchy
          for id in range(1, maximal_num_hierarchies):
            priority_queue[id].clear()
          # record if better noise
          loss = losses[0]
          if loss < best_loss:
            best_noise = curr_noise
            best_loss = loss

        if i_run > 1:
          _, idx = heapq.heappop(priority_queue[0])
          curr_block = blocks[0][idx//2]
          if i > 0:
            tf.logging.info("After {} initial blocks, num queries: {}, loss: {:.4f}".format(i, num_queries, curr_loss))

        ## Refining starting from second hierarchy
        stack = []
        root = curr_block  # restart from the root node
        stack.append(root)
        noise, loss, queries, success = self.node.perturb(root, blocks, priority_queue, curr_noise, best_loss, label, image, sess, skip)
        num_queries += queries
        
        if success:
          tf.logging.info("num queries: {}".format(num_queries))
          return adv_image, curr_noise, num_queries, True, flag
        # If query count exceeds the maximum queries, then return False
        if num_queries > self.max_queries:
          tf.logging.info("num queries: {}".format(num_queries))
          return self._perturb_image(image, noise), noise, num_queries, False, flag
          
        if loss >= best_loss:
          curr_noise = best_noise  # restore to the previous noise
          continue
        else:
          curr_loss = loss
          curr_noise = noise
          adv_image = self._perturb_image(image, curr_noise)
          best_loss = curr_loss
          best_noise = curr_noise

        skip_flag1 = skip
        skip_flag2 = skip
        while len(stack) > 0:
          cur = stack.pop()
          if skip_flag2:
            j = self.node.get_idx(cur) + skip
            skip_flag2 = 0
          else:
            j = self.node.get_idx(cur)
          if priority_queue[j] and priority_queue[j][0][0] < 0:  # if the margin is better
            if j < maximal_num_hierarchies - 1:
              if skip_flag1:
                next = self.node.get_next_node(cur, priority_queue, blocks, skip)
                skip_flag1 = 0
              else:
                next = self.node.get_next_node(cur, priority_queue, blocks)
              # keep the depth-first order
              stack.append(cur)
              stack.append(next)

              # operate on the next node
              noise, loss, queries, success = self.node.perturb(next, blocks, priority_queue, curr_noise, best_loss, label, image, sess)
              num_queries += queries
              if loss < best_loss or success:
                curr_loss = loss
                curr_noise = noise
                adv_image = self._perturb_image(image, curr_noise)
                best_loss = curr_loss
                best_noise = curr_noise
                if success:
                  tf.logging.info("num queries: {}".format(num_queries))
                  return adv_image, curr_noise, num_queries, True, flag
                # If query count exceeds the maximum queries, then return False
                if num_queries > self.max_queries:
                  tf.logging.info("num queries: {}".format(num_queries))
                  return adv_image, curr_noise, num_queries, False, flag
              else:  # if the loss isn't better, give up the remaining blocks in the priority queue
                stack.pop()
                priority_queue[self.node.get_idx(next)].clear()
                heapq.heappop(priority_queue[j])
            else:
              heapq.heappop(priority_queue[j])

      priority_queue[0] = rec_queue.copy()
      i_run += 1

      if skip < maximal_skip_num:
        skip += 1
        if self.block_size/pow(2, skip) <= self.min_block_size:
          break
      else:
        break

    # If Saliency Attack has exhausted, then return False
    if num_queries <= self.max_queries:
      tf.logging.info("num queries: {}".format(num_queries))
      return adv_image, curr_noise, num_queries, False, flag
