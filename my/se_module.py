# # Squeeze and Excitation phase
#   if 0 < se_ratio <= 1:
#     filters_se = max(1, int(filters_in * se_ratio))
#     se = layers.GlobalAveragePooling2D(name=name + 'se_squeeze')(x)
#     se = layers.Reshape((1, 1, filters), name=name + 'se_reshape')(se)
#     se = layers.Conv2D(
#         filters_se,
#         1,
#         padding='same',
#         activation=activation,
#         kernel_initializer=CONV_KERNEL_INITIALIZER,
#         name=name + 'se_reduce')(
#             se)
#     se = layers.Conv2D(
#         filters,
#         1,
#         padding='same',
#         activation='sigmoid',
#         kernel_initializer=CONV_KERNEL_INITIALIZER,
#         name=name + 'se_expand')(se)
#     x = layers.multiply([x, se], name=name + 'se_excite')
