import heapq
import math
import cv2
import sys
import numpy as np
import tensorflow as tf

class Node(object):
  """A helper for SaliencyAttack algorithm to represent each node in a tree.
  """

  def __init__(self, model, args, **kwargs):
    """Initalize local search helper.

    Args:
      model: TensorFlow model
      args: arguments
    """
    self.block_size = args.block_size  # int, initial block size
    self.loss_func = args.loss_func

    # Network setting
    self.x_input = model['x_input']
    self.y_input = model['y_input']
    self.logits = model['logits']
    self.preds = model['preds']

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

  def _filter_finer_block(self, blocks_all, coarse_block):
    """choose the finer blocks in the coarse_block region

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

  def _perturb_image(self, image, noise):
    """Given an image and a noise, generate a perturbed image.
    First, resize the noise with the size of the image.
    Then, add the resized noise to the image.

    Args:
      image: numpy array of size [1, 299, 299, 3], an original image
      noise: numpy array of size [1, 256, 256, 3], a noise

    Returns:
      adv_image: numpy array of size [1, 299, 299, 3], a perturbed image
    """
    adv_image = image + cv2.resize(noise[0, ...], (self.width, self.height), interpolation=cv2.INTER_NEAREST)
    adv_image = np.clip(adv_image, 0., 1.)
    return adv_image

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

  def get_idx(self, block):
    """
    get the index of the next hierarchy

    Args:
      block: list, the current block
    """
    curr_block = block
    ul, lr, _, _ = curr_block
    curr_block_size = abs(lr[0] - ul[0])
    i = math.log(self.block_size, 2) - math.log(curr_block_size, 2) + 1

    return int(i)

  def get_next_node(self, block, priority_queue, blocks, skip=0):
    """
    get the next finer block with the best loss in the priority queue

    Args:
      blocks: 2D list, the set of blocks in salient region
      priority_queue: 2D list, the priority queue for min heap, recording the index of the best block in all hierarchies
      block: list, the current block
      skip: int, default 0 means do not skip any hierarchy so that just perturb the finer blocks in the next hierarchy
    """
    i = self.get_idx(block) + skip
    curr_block = block
    finer_blocks = self._filter_finer_block(blocks[i], curr_block)  # filter finer blocks in current block region
    margin, idx = heapq.heappop(priority_queue[i])
    next_block = finer_blocks[idx]

    # To avoid the secondary pop-up of priority queue in perturb function
    heapq.heappush(priority_queue[i], (margin, idx))
    return next_block

  def perturb(self, block, blocks, priority_queue, curr_noise, best_loss, label, image, sess, skip=0):
    """
    test the finer blocks and update the priority queue

    Args:
      block: list, the current block
      blocks: 2D list, the set of blocks in salient region
      prioirty_queue: 2D list, the priority queue for min heap, recording the index of the best block in all hierarchies
      curr_noise: numpy array of size [1, 256, 256, 3], current noise
      best_loss: float, best loss
      label: numpy array of size [1], the label of the image (or target label)
      image: numpy array of size [1, 299, 299, 3], an original image
      sess: TensorFlow session
      skip: int, default 0 means do not skip any hierarchy so that just perturb the finer blocks in the next hierarchy
    """

    self.width = image.shape[1]
    self.height = image.shape[2]
    i = self.get_idx(block) + skip   # get the index of the next hierarchy
    queries = 0

    # log the Refining process
    # ul, lr, _, _ = block
    # curr_block_size = abs(lr[0] - ul[0])
    # tf.logging.info("block_size: {}, test on {}th hierarchy".format(curr_block_size, i))

    # restrict perturbation only in curr_block region
    finer_blocks = self._filter_finer_block(blocks[i], block)
    num_blocks = len(finer_blocks)

    # if the priority_queue for the next hierarchy exist, we just choose the best block to test
    if priority_queue[i]:
      _, idx = heapq.heappop(priority_queue[i])
      finer_block = finer_blocks[idx]

      noise = self._flip_noise(curr_noise, finer_block)
      image_test = self._perturb_image(image, noise)
      losses, pred = sess.run([self.losses, self.preds],
        feed_dict={self.x_input: image_test, self.y_input: label})
      queries += 1
      loss = losses[0]
      # Early stopping
      flag = (pred != label)
      if flag:
        return noise, loss, queries, True

    # when the priority_queue for the next hierarchy is empty, we test all finer blocks and update the priority_queue
    else:
      # test noises added in these blocks at ith hierarchy split
      if num_blocks > 100:
        noise = np.copy(curr_noise)
        loss = 1.0
        batch_size = 100
        losses = np.zeros(num_blocks, dtype=np.float32)
        preds = np.zeros(num_blocks, dtype=np.int64)
        num_batches = int(math.ceil(num_blocks / batch_size))
        for ibatch in range(num_batches):
          bstart = ibatch * batch_size
          if (num_blocks-bstart) < batch_size:
            batch_size = num_blocks - bstart
          bend = bstart + batch_size
          image_batch = np.zeros([batch_size, self.width, self.height, 3], np.float32)
          noise_batch = np.zeros([batch_size, 256, 256, 3], np.float32)
          label_batch = np.tile(label, batch_size)
          for k in range(batch_size):
            # flip the noise
            noise_batch[k] = self._flip_noise(curr_noise, finer_blocks[k+bstart])
            image_batch[k] = self._perturb_image(image, noise_batch[k:k + 1])
          losses[bstart:bend], preds[bstart:bend] = sess.run([self.losses, self.preds],
            feed_dict={self.x_input: image_batch, self.y_input: label_batch})

          # Early stopping
          success_indices, = np.where(preds[bstart:bend] != label)
          if len(success_indices) > 0:
            curr_noise[0, ...] = noise_batch[success_indices[0], ...]
            curr_loss = losses[bstart+success_indices[0]]
            queries += bstart + success_indices[0] + 1
            return curr_noise, curr_loss, queries, True

          queries += batch_size

          # Push the margin and the corresponding index into the priority queue
          for k in range(batch_size):
            heapq.heappush(priority_queue[i], (losses[bstart+k] - best_loss, bstart+k))

          # choose the current best loss and update the noise
          margin, idx = heapq.heappop(priority_queue[i])
          if margin+best_loss < loss:
            noise = noise_batch[idx-bstart:idx-bstart + 1]
            heapq.heappush(priority_queue[i], (margin, idx))
            loss = margin + best_loss
      else:
        image_batch = np.zeros([num_blocks, self.width, self.height, 3], np.float32)
        noise_batch = np.zeros([num_blocks, 256, 256, 3], np.float32)
        label_batch = np.tile(label, num_blocks)
        for k in range(num_blocks):
          # flip the noise
          noise_batch[k] = self._flip_noise(curr_noise, finer_blocks[k])
          image_batch[k] = self._perturb_image(image, noise_batch[k:k + 1])
        losses, preds = sess.run([self.losses, self.preds],
            feed_dict={self.x_input: image_batch, self.y_input: label_batch})

        # Early stopping
        success_indices, = np.where(preds != label)
        if len(success_indices) > 0:
          curr_noise[0, ...] = noise_batch[success_indices[0], ...]
          curr_loss = losses[success_indices[0]]
          queries += success_indices[0] + 1
          return curr_noise, curr_loss, queries, True

        queries += num_blocks

        # Push the margin and the corresponding index into the priority queue
        for k in range(num_blocks):
          heapq.heappush(priority_queue[i], (losses[k] - best_loss, k))

        # choose the current best loss and update the noise
        margin, idx = heapq.heappop(priority_queue[i])
        noise = noise_batch[idx:idx + 1]
        heapq.heappush(priority_queue[i], (margin, idx))
        loss = margin + best_loss

    return noise, loss, queries, False
