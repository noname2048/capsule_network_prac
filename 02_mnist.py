import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/")

n_samples = 5
plt.figure(figsize=(n_samples * 1.5, 3)) # ratio of figure box
for index in range(n_samples):
    plt.subplot(1, n_samples, index + 1)
    sample_image = mnist.train.images[index].reshape(28, 28)
    plt.title(mnist.train.labels[index], loc="center")
    plt.imshow(sample_image, cmap="binary")

plt.show()
