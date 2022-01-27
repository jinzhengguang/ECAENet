import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

# 2021-04-10 guangjinzheng
import tensorflow as tf
from tensorflow.python.keras.layers import Activation
from tensorflow.keras.utils import get_custom_objects
class Hswish(Activation):
    # X = Activation('hswish', name="conv1_act")(X_input)
    def __init__(self, activation, **kwargs):
        super(Hswish, self).__init__(activation, **kwargs)
        self.__name__ = 'hswish'

def h_swish(inputs):
    # return tf.nn.relu(inputs)   # relu
    # return tf.nn.swish(inputs)    # swish
    return inputs * tf.nn.relu6(inputs + 3.0) / 6.0  # h-swish

get_custom_objects().update({'hswish': Hswish(h_swish)})
# 2021-04-10 guangjinzheng

# my activate function
class myaf(object):
    def __init__(self, x=[-2.1, 0, 2.14]):
        self.x = x
        self.input = tf.constant(self.x)

    def relu(self):
        y = tf.nn.relu(self.input)
        return y.numpy()

    def swish(self):
        y = tf.nn.swish(self.input)
        return y.numpy()

    def hswish(self):
        y = self.input * tf.nn.relu6(self.input + 3.0) / 6.0
        return y.numpy()

    def hswish_test(self):
        y = tf.keras.layers.Activation('hswish', name="conv_hswish")(self.input)
        print(self.hswish())
        print(y.numpy())
        return y.numpy()

    def test(self):
        print(self.relu())
        print(self.swish())
        print(self.hswish())

    def plot(self):
        sz = 14
        plt.figure(dpi=150)
        plt.xlabel('x', fontsize=sz)
        plt.ylabel('f(x)', fontsize=sz)
        plt.title('Activate Function')
        plt.xlim([-7, 7])
        plt.ylim([-1, 7])
        plt.gca().xaxis.set_major_locator(MultipleLocator(2))
        plt.gca().yaxis.set_major_locator(MultipleLocator(1))
        plt.plot(self.x, self.relu(), label='relu')
        plt.plot(self.x, self.swish(), label='swish')
        plt.plot(self.x, self.hswish(), label='hswish')
        # plt.plot(self.x, self.relu(), self.x, self.swish(), self.x, self.hswish())
        plt.legend()
        plt.show()

    def sheet(self):
        y1 = self.relu()
        y2 = self.swish()
        y3 = self.hswish()
        print('x y_relu y_swish y_hswish')
        for i in range(len(self.x)):
            print('{:.6f} {:.6f} {:.6f} {:.6f}'.format(self.x[i], y1[i], y2[i], y3[i]))

if __name__ == '__main__':
    x = np.linspace(-6, 6, 100)
    f = myaf(x)
    f.hswish_test()
    # f.sheet()
    f.plot()

# 2021-04-10 guangjinzheng activate function
