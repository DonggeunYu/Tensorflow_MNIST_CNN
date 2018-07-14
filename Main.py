import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def weight_variable(shape):
    initial = tf.Variable(tf.truncated_normal(shape=shape, stddev=0.1))
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.Variable(tf.truncated_normal(shape=shape, stddev=0.1))
    return tf.Variable(initial)


def conv2d(x, w, bias):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME') + bias


def relu(x):
    return tf.nn.relu(x)


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# 데이터 셋 블러오기
mnist = input_data.read_data_sets("data/MNIST_data/", one_hot=True, reshape=False)

X = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
Y_Label = tf.placeholder(tf.float32, shape=[None, 10])

Conv1 = conv2d(X, weight_variable([4, 4, 1, 4]), bias_variable([4]))
Relu1 = relu(Conv1)
Pool1 = max_pool_2x2(Relu1)

Conv2 = conv2d(Pool1, weight_variable([4, 4, 4, 8]), bias_variable([8]))
Relu2 = relu(Conv2)
Pool2 = max_pool_2x2(Relu2)


W1 = tf.Variable(tf.truncated_normal(shape=[8*7*7, 10]))
b1 = tf.Variable(tf.truncated_normal(shape=[10]))
Pool2_flat = tf.reshape(Pool2, [-1, 8*7*7])
OutputLayer = tf.matmul(Pool2_flat, W1) + b1

Loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y_Label, logits=OutputLayer))
train_step = tf.train.AdamOptimizer(0.005).minimize(Loss)

correct_prediction = tf.equal(tf.arg_max(OutputLayer, 1), tf.arg_max(Y_Label, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(1000):
        trainingData, Y = mnist.train.next_batch(64)
        sess.run(train_step, feed_dict={X: trainingData, Y_Label: Y})
        if i % 100 == 0:
            print(sess.run(accuracy, feed_dict={X: mnist.test.images, Y_Label: mnist.test.labels}))