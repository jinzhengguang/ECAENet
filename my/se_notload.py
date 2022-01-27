#     # 2021-03-30 guangjinzheng
#     # for i in model.layers:
#     #         if i.name in [j.name for j in modelpre.layers]:
#     #             temp = modelpre.get_layer(i.name).get_weights()
#     #             i.set_weights(temp)
#     # model.summary()
#     for layeri in model0.layers:
#         if 'se_' in layeri.name:
#             print(layeri.name)
#         else:
#             layeri.set_weights(model.get_layer(layeri.name).get_weights())
#     return model0
