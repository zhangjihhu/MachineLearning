import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import np_utils


class SoftmaxRegression:

    def relu(self, x):
        return np.maximum(0, x)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def softmax(self, x):
        e = np.exp(x)
        return e / np.sum(e, axis=1, keepdims=True)

    def normalization(self, x):
        max = np.max(x, axis=0)
        min = np.max(x, axis=0)
        return (x - min) / (max -min)

    def param_init(self, m, k):
        w = 0.1 * np.random.randn(m, k)
        b = np.zeros((1, k))
        return w, b

    def train(self, x, y, epochs, lr):
        num_feature = x.shape[1]
        num_train = x.shape[0]
        w1, b1 = self.param_init(num_feature, 20)
        w2, b2 = self.param_init(20, 10)
        loss_list = []
        for i in range(epochs):
            net1 = np.dot(x, w1) + b1
            out1 = self.sigmoid(net1)
            net2 = np.dot(out1, w2) + b2
            softmax = self.softmax(net2)
            dnet2 = softmax - y
            dw2 = np.dot(out1.T, dnet2) / num_train
            db2 = np.sum(dnet2, axis=0, keepdims=True) / num_train
            w2 -= lr * dw2
            b2 -= lr * db2
            dnet1 = np.dot(dnet2, w2.T) * out1 * (1 - out1)
            dw1 = np.dot(x.T, dnet1) / num_train
            db1 = np.sum(dnet1, axis=0, keepdims=True) / num_train
            w1 -= lr * dw1
            b1 -= lr * db1
            loss = -1 * np.sum(y * np.log(softmax)) / num_train
            loss_list.append(loss)
            print('epochs: %d  loss:  %f' % (i, loss))
        params = {'w1': w1, 'b1': b1,
                  'w2': w2, 'b2': b2}
        return params, loss_list

    def predict(self, x, params):
        w1, b1, w2, b2 = params['w1'], params['b1'], params['w2'], params['b2']
        net1 = np.dot(x, w1) + b1
        out1 = self.sigmoid(net1)
        net2 = np.dot(out1, w2) + b2
        softmax = self.softmax(net2)
        return softmax

    def props_to_onehot(self, props):
        a = np.argmax(props, axis=1)
        b = np.zeros((len(a), props.shape[1]))
        b[np.arange(len(a)), a] = 1
        return b

    def accuracy(self, y_test, y_pred):
        return np.mean(np.equal(y_test, y_pred))

    def confusion_matrix(self, y_test, y_pred, n_class):
        matrix = np.zeros((n_class, n_class))
        labels = np.unique(y_test).T
        for i in labels:
            idx = np.where(y_test == i)
            pred = y_pred[idx]
            for j in labels:
                val = len(np.where(pred == j)[0])
                matrix[int(i), int(j)] = val
        return matrix


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    pixels = x_train.shape[1] * x_train.shape[2]
    x_train = x_train.reshape(x_train.shape[0], pixels).astype('float32') / 255
    x_test = x_test.reshape(x_test.shape[0], pixels).astype('float32') / 255
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)

    soft = SoftmaxRegression()
    params, loss_list = soft.train(x_train, y_train, 1000, 0.1)

    y_pred = soft.predict(x_test, params)
    y_pred = soft.props_to_onehot(y_pred)
    accuracy = soft.accuracy(y_test, y_pred)
    print('accuracy: ', accuracy)
    confusion_matrix = soft.confusion_matrix(y_test, y_pred, 10)
    print('confusion matrix:', confusion_matrix)