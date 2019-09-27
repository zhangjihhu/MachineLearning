import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as datasets
import tensorflow as tf


def param_init(n, m):
    w = 0.1 * np.random.randn(n, m)
    b = np.zeros((1, m))
    return w, b


def normalization(x):
    max = np.max(x, axis=0)
    min = np.min(x, axis=0)
    return (x - min) / (max - min), max, min


boston = datasets.load_boston()
x, x_min, x_max = normalization(np.array(boston.data))
y, y_min, y_max = normalization(np.array(boston.target))
y = y.reshape(-1, 1)

offset = int(x.shape[0] * 0.8)
x_train = x[:offset]
y_trian = y[:offset]
x_test = x[offset:]
y_test = y[offset:]

x = tf.placeholder(tf.float32, [None, x.shape[1]])
y = tf.placeholder(tf.float32, [None, 1])
w, b = param_init(x.shape[1], 1)
w = tf.Variable(w)
b = tf.Variable(b)
y_hat = tf.matmul(x, w) + b
loss = tf.reduce_sum(tf.square(y - y_hat))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for i in range(1000):
    sess.run(train_step, feed_dict={x: x_train, y: y_trian})





