import tensorflow as tf
import numpy as np
import math

# my attention
def myeca(x):
    channels = tf.keras.backend.int_shape(x)[-1]
    gamma = 2
    b = 1
    t = int(abs((math.log(channels, 2) + b) / gamma))
    k = t if t % 2 else t + 1
    squeeze = tf.reduce_mean(x, [2, 3], keepdims=False)
    squeeze = tf.expand_dims(squeeze, axis=1)
    attn = tf.keras.layers.Conv1D(filters=1, kernel_size=k, padding='same', use_bias=False)(squeeze)
    attn = tf.expand_dims(tf.transpose(attn, [0, 2, 1]), 3)
    attn = tf.math.sigmoid(attn)
    scale = x * attn
    x = scale
    return x

if __name__ == '__main__':
    a = np.asarray([22, 23, 5])
    b = tf.keras.backend.constant(a)
    c = tf.keras.backend.eval(b)

    img_input = tf.keras.layers.Input(shape=(112, 112, 3))
    x = myeca(img_input)
    print(x)

# 2021-04-10 guangjinzheng my attention
