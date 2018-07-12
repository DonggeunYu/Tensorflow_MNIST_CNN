import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("data/MNIST_data/", one_hot=False)

x, y = mnist.train.next_batch(1, shuffle=False)
train_image_x_1d = np.array(x)
train_image_x_2d = train_image_x_1d.reshape((28, 28, -1))
np.array(train_image_x_2d, dtype=np.float32)
print(train_image_x_2d.shape)
plt.imshow(train_image_x_2d.reshape(28, 28), cmap='Greys')
plt.show()