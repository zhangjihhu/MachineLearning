import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class LogisticRegression:

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def param_init(self, n, m):
        w = 0.01 * np.random.randn(n, m)
        b = np.zeros((1, m))
        return w, b

    def normalization(self, x):
        min = np.min(x, axis=0)
        max = np.max(x, axis=0)
        return (x - min) / (max - min), max, min

    def train(self, x, y, epochs, lr):
        num_train = x.shape[0]
        num_feature = x.shape[1]
        w, b = self.param_init(num_feature, 1)
        loss_list = []
        for i in range(epochs):
            net = np.dot(x, w) + b
            out = self.sigmoid(net)
            loss = -1 * np.sum(y * np.log(out) + (1 - y) * np.log(1 - out)) / num_train
            print('epochs %d loss %f' % (i, loss))
            loss_list.append(loss)
            dnet = out - y
            dw = np.dot(x.T, dnet) / num_train
            db = np.mean(dnet, axis=0)
            w -= lr * dw
            b -= lr * db
        param = {'w': w, 'b': b}
        return param, loss_list

    def predict(self, x, params):
        w, b = params['w'], params['b']
        y_pred = self.sigmoid(np.dot(x, w) + b)
        for i in range(y_pred.shape[0]):
            if y_pred[i] >= 0.5:
                y_pred[i] = 1
            else:
                y_pred[i] = 0
        return y_pred

    def accuracy(self, y_test, y_pred):
        return np.mean(np.equal(y_test, y_pred))

    def confusion_matrix(self, y_test, y_pred, n_class):
        matrix = np.zeros((n_class, n_class))
        labels = np.unique(y_test)
        labels = labels.T
        for i in labels:
            idx = np.where(y_test == i)
            pred = y_pred[idx]
            for j in labels:
                val = len(np.where(pred == j)[0])
                matrix[int(i), int(j)] = val
        return matrix


if __name__ == '__main__':
    logistic = LogisticRegression()
    pd_data = np.array(pd.read_csv('./HTRU_2.csv'))
    np.random.shuffle(pd_data)
    x, x_min, x_max = logistic.normalization(pd_data[:, :8])
    y = pd_data[:, 8:]
    offset = int(x.shape[0] * 0.999)
    x_train = x[: offset]
    y_train = y[: offset]
    x_test = x[offset:]
    y_test = y[offset:]

    params, loss_list = logistic.train(x_train, y_train, 10000, 0.1)

    y_pred = logistic.predict(x_test, params)
    print('accuracy:', logistic.accuracy(y_test, y_pred))
    print('confusion matrix: ', logistic.confusion_matrix(y_test, y_pred, 2))

    plt.plot(loss_list)
    plt.show()

    x_index = np.arange(0, x_test.shape[0], 1)
    plt.scatter(x_index, y_test, color='black')
    plt.show()
    plt.scatter(x_index, y_pred, color='red')
    plt.show()

