import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as datasets


class LinerRegression:

    def neure(self, x, w, b):
        return np.dot(x, w) + b

    def param_init(self, n, m):
        w = 0.1 * np.random.randn(n, m)
        b = np.zeros((1, m))
        return w, b

    def normalization(self, x):
        max = np.max(x, axis=0)
        min = np.min(x, axis=0)
        return (x - min) / (max - min), max, min

    def train(self, x, y, epochs, lr):
        num_train = x.shape[0]
        num_feature = x.shape[1]
        w, b = self.param_init(x.shape[1], 1)
        loss_list = []
        for i in range(epochs):
            net = self.neure(x, w, b)
            out = net
            dnet = out - y
            loss = np.sum(np.square(dnet)) / num_train
            print('epochs %d loss %f' % (i, loss))
            loss_list.append(loss)
            dw = np.dot(x.T, dnet) / num_train
            db = np.mean(dnet, axis=0) / num_train
            w -= lr * dw
            b -= lr * db
        params = {'w': w, 'b': b}
        return params, loss_list

    def predict(self, x, w, b):
        return self.neure(x, w, b)


if __name__ == '__main__':
        liner = LinerRegression()
        boston = datasets.load_boston()
        x, x_min, x_max = liner.normalization(np.array(boston.data))
        y, y_min, y_max = liner.normalization(np.array(boston.target))
        y = y.reshape(-1, 1)

        offset = int(x.shape[0] * 0.8)
        x_train = x[:offset]
        y_trian = y[:offset]
        x_test = x[offset:]
        y_test = y[offset:]

        w, b = liner.param_init(x_train.shape[1], 1)
        param, loss_list = liner.train(x_train, y_trian, 10000, 0.01)
        y_pred = liner.predict(x_test, param['w'], param['b'])

        plt.plot(loss_list)
        plt.show()

        x_index = np.arange(0, y_test.shape[0], 1)
        plt.plot(x_index, y_test, color='black')
        plt.plot(x_index, y_pred, color='red')
        plt.show()

