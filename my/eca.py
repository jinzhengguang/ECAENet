# if has_se:
#     # 2021-04-08 guangjinzheng
#     import tensorflow as tf
#     import keras
#     import keras.backend as K
#     channels = K.int_shape(x)[-1]
#     gamma = 2
#     b = 1
#     t = int(abs((math.log(channels, 2) + b) / gamma))
#     k = t if t % 2 else t + 1
#     squeeze = tf.reduce_mean(x, [2, 3], keepdims=False)
#     squeeze = tf.expand_dims(squeeze, axis=1)
#     attn = keras.layers.Conv1D(filters=1, kernel_size=k, padding='same', use_bias=False)(squeeze)
#     attn = tf.expand_dims(tf.transpose(attn, [0, 2, 1]), 3)
#     attn = tf.math.sigmoid(attn)
#     scale = x * attn
#     x = scale
