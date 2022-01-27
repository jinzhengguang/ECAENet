from tensorflow.python.keras.applications.vgg16 import VGG16
from tensorflow.python.keras.applications.resnet_v2 import ResNet101V2
from tensorflow.python.keras.applications.inception_v3 import InceptionV3
from tensorflow.python.keras.applications.xception import Xception
from tensorflow.python.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.python.keras.applications.mobilenet_v3 import MobileNetV3Small
from tensorflow.python.keras.applications.densenet import DenseNet201
from tensorflow.python.keras.applications.densenet import DenseNet169
from tensorflow.python.keras.applications.nasnet import NASNetMobile
import tensorflow.python.keras.applications.efficientnet as efn
from tensorflow.python.keras import Model
import tensorflow.keras.layers as layers
import applications.efficientnet as eca
import af   # hswish

# AlexNet
def myAlexNet(input_shape=(224, 224, 3), classes=1000):
    inputs = layers.Input(shape=input_shape)
    c1 = layers.Conv2D(48, (11, 11), strides=4, activation='relu', kernel_initializer='uniform', padding='valid')(inputs)
    c2 = layers.BatchNormalization()(c1)
    c3 = layers.MaxPool2D((3, 3), strides=2, padding='valid')(c2)
    c4 = layers.Conv2D(128, (5, 5), strides=1, padding='same', activation='relu', kernel_initializer='uniform')(c3)
    c5 = layers.BatchNormalization()(c4)
    c6 = layers.MaxPool2D((3, 3), strides=2, padding='valid')(c5)
    c7 = layers.Conv2D(192, (3, 3), strides=1, padding='same', activation='relu', kernel_initializer='uniform')(c6)
    c8 = layers.Conv2D(192, (3, 3), strides=1, padding='same', activation='relu', kernel_initializer='uniform')(c7)
    c9 = layers.Conv2D(128, (3, 3), strides=1, padding='same', activation='relu', kernel_initializer='uniform')(c8)
    c10 = layers.MaxPool2D((3, 3), strides=2, padding='valid')(c9)
    c11 = layers.Flatten()(c10)
    c12 = layers.Dense(2048, activation='relu')(c11)
    c13 = layers.Dropout(0.3)(c12)
    c14 = layers.Dense(2048, activation='relu')(c13)
    c15 = layers.Dropout(0.3)(c14)
    outputs = layers.Dense(classes, activation='softmax')(c15)
    model = Model(inputs=[inputs], outputs=[outputs])
    model.summary()
    return model

# VGG16
def myVGG16(input_shape=(224, 224, 3), classes=1000):
    pre_trained_model = VGG16(input_shape=input_shape, weights='imagenet', include_top=False)
    # pre_trained_model = VGG16(weights='imagenet', include_top=True)
    # pre_trained_model.summary()
    x = pre_trained_model.output
    x = layers.Flatten(name='flatten')(x)
    x = layers.Dense(4096, activation='relu', name='fc1')(x)
    x = layers.Dense(4096, activation='relu', name='fc2')(x)
    predictions = layers.Dense(classes, activation='softmax')(x)
    model = Model(inputs=pre_trained_model.input, outputs=predictions)
    model.summary()
    return model

# ResNetV2101
def myResNetV2101(input_shape=(224, 224, 3), classes=1000):
    pre_trained_model = ResNet101V2(input_shape=input_shape, weights='imagenet', include_top=False)
    # pre_trained_model = ResNet101V2(weights='imagenet', include_top=True)
    # pre_trained_model.summary()
    x = pre_trained_model.output
    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    predictions = layers.Dense(classes, activation='softmax')(x)
    model = Model(inputs=pre_trained_model.input, outputs=predictions)
    model.summary()
    return model

# InceptionV3
def myInceptionV3(input_shape=(299, 299, 3), classes=1000):
    pre_trained_model = InceptionV3(input_shape=input_shape, weights='imagenet', include_top=False)
    # pre_trained_model = InceptionV3(weights='imagenet', include_top=True)
    # pre_trained_model.summary()
    x = pre_trained_model.output
    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    predictions = layers.Dense(classes, activation='softmax')(x)
    model = Model(inputs=pre_trained_model.input, outputs=predictions)
    model.summary()
    return model

# Xception
def myXception(input_shape=(299, 299, 3), classes=1000):
    pre_trained_model = Xception(input_shape=input_shape, weights='imagenet', include_top=False)
    # pre_trained_model = Xception(weights='imagenet', include_top=True)
    # pre_trained_model.summary()
    x = pre_trained_model.output
    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    predictions = layers.Dense(classes, activation='softmax')(x)
    model = Model(inputs=pre_trained_model.input, outputs=predictions)
    model.summary()
    return model

# MobileNetV2
def myMobileNetV2(input_shape=(224, 224, 3), classes=1000):
    pre_trained_model = MobileNetV2(input_shape=input_shape, weights='imagenet', include_top=False)
    # pre_trained_model = MobileNetV2(weights='imagenet', include_top=True)
    # pre_trained_model.summary()
    x = pre_trained_model.output
    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    predictions = layers.Dense(classes, activation='softmax')(x)
    model = Model(inputs=pre_trained_model.input, outputs=predictions)
    model.summary()
    return model

# MobileNetV3Small
def myMobileNetV3Small(input_shape=(224, 224, 3), classes=1000):
    # pre_trained_model = MobileNetV3Small(input_shape=input_shape, weights='imagenet', include_top=True)
    # pre_trained_model.summary()
    pre_trained_model = MobileNetV3Small(input_shape=input_shape, weights='imagenet', include_top=False)
    x = pre_trained_model.output
    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    x = layers.Dropout(0.2)(x)
    predictions = layers.Dense(classes, activation='softmax')(x)
    model = Model(inputs=pre_trained_model.input, outputs=predictions)
    model.summary()
    return model

# DenseNet201
def myDenseNet201(input_shape=(224, 224, 3), classes=1000):
    pre_trained_model = DenseNet201(input_shape=input_shape, weights='imagenet', include_top=False)
    # pre_trained_model = DenseNet201(weights='imagenet', include_top=True)
    # pre_trained_model.summary()
    x = pre_trained_model.output
    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    predictions = layers.Dense(classes, activation='softmax')(x)
    model = Model(inputs=pre_trained_model.input, outputs=predictions)
    model.summary()
    return model

# DenseNet169
def myDenseNet169(input_shape=(224, 224, 3), classes=1000):
    pre_trained_model = DenseNet169(input_shape=input_shape, weights='imagenet', include_top=False)
    # pre_trained_model = DenseNet169(weights='imagenet', include_top=True)
    # pre_trained_model.summary()
    x = pre_trained_model.output
    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    predictions = layers.Dense(classes, activation='softmax')(x)
    model = Model(inputs=pre_trained_model.input, outputs=predictions)
    model.summary()
    return model

# NASNetMobile
def myNASNetMobile(input_shape=(224, 224, 3), classes=1000):
    pre_trained_model = NASNetMobile(input_shape=input_shape, weights='imagenet', include_top=False)
    # pre_trained_model = NASNetMobile(weights='imagenet', include_top=True)
    # pre_trained_model.summary()
    x = pre_trained_model.output
    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    predictions = layers.Dense(classes, activation='softmax')(x)
    model = Model(inputs=pre_trained_model.input, outputs=predictions)
    model.summary()
    return model

# EfficientNetB0
def myEfficientNetB0(input_shape=(224, 224, 3), classes=1000):
    pre_trained_model = efn.EfficientNetB0(input_shape=input_shape, weights='imagenet', include_top=False)
    # pre_trained_model = ef.EfficientNetB0(weights='imagenet', include_top=True)
    # pre_trained_model.summary()
    x = pre_trained_model.output
    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    x = layers.Dropout(0.2, name='top_dropout')(x)
    predictions = layers.Dense(classes, activation='softmax')(x)
    model = Model(inputs=pre_trained_model.input, outputs=predictions)
    # for layer in model.layers[:-6]:
    #     layer.trainable = False
    #     print(layer.name)
    model.summary()
    return model

# EfficientNet
def myEfficientNet(model_str='EfficientNetB0', attention='se', activation='swish', input_shape=(224, 224, 3), classes=1000):
    # 2021-04-26 guangjinzheng
    top_dropout = 0.2
    pre_trained_model = ''
    if model_str in 'EfficientNetB0':
        pre_trained_model = efn.EfficientNetB0(activation=activation, input_shape=input_shape, weights='imagenet', include_top=False)
    if model_str in 'EfficientNetB1':
        top_dropout = 0.2
        pre_trained_model = efn.EfficientNetB1(activation=activation, input_shape=(240, 240, 3), weights='imagenet', include_top=False)
    elif model_str in 'EfficientNetB2':
        top_dropout = 0.3
        pre_trained_model = efn.EfficientNetB2(activation=activation, input_shape=(260, 260, 3), weights='imagenet', include_top=False)
    elif model_str in 'EfficientNetB3':
        top_dropout = 0.3
        pre_trained_model = efn.EfficientNetB3(activation=activation, input_shape=(300, 300, 3), weights='imagenet', include_top=False)
    elif model_str in 'EfficientNetB4':
        top_dropout = 0.4
        pre_trained_model = efn.EfficientNetB4(activation=activation, input_shape=(380, 380, 3), weights='imagenet', include_top=False)
    elif model_str in 'EfficientNetB5':
        top_dropout = 0.4
        pre_trained_model = efn.EfficientNetB5(activation=activation, input_shape=(456, 456, 3), weights='imagenet', include_top=False)
    elif model_str in 'EfficientNetB6':
        top_dropout = 0.5
        pre_trained_model = efn.EfficientNetB6(activation=activation, input_shape=(528, 528, 3), weights='imagenet', include_top=False)
    elif model_str in 'EfficientNetB7':
        top_dropout = 0.5
        pre_trained_model = efn.EfficientNetB7(activation=activation, input_shape=(600, 600, 3), weights='imagenet', include_top=False)
    else:
        pass

    if attention is not 'se':
        my_model = ''
        if model_str in 'EfficientNetB0':
            my_model = eca.EfficientNetB0(activation=activation, input_shape=input_shape, weights=None, include_top=False)
        if model_str in 'EfficientNetB1':
            my_model = eca.EfficientNetB1(activation=activation, input_shape=(240, 240, 3), weights=None, include_top=False)
        elif model_str in 'EfficientNetB2':
            my_model = eca.EfficientNetB2(activation=activation, input_shape=(260, 260, 3), weights=None, include_top=False)
        elif model_str in 'EfficientNetB3':
            my_model = eca.EfficientNetB3(activation=activation, input_shape=(300, 300, 3), weights=None, include_top=False)
        elif model_str in 'EfficientNetB4':
            my_model = eca.EfficientNetB4(activation=activation, input_shape=(380, 380, 3), weights=None, include_top=False)
        elif model_str in 'EfficientNetB5':
            my_model = eca.EfficientNetB5(activation=activation, input_shape=(456, 456, 3), weights=None, include_top=False)
        elif model_str in 'EfficientNetB6':
            my_model = eca.EfficientNetB6(activation=activation, input_shape=(528, 528, 3), weights=None, include_top=False)
        elif model_str in 'EfficientNetB7':
            my_model = eca.EfficientNetB7(activation=activation, input_shape=(600, 600, 3), weights=None, include_top=False)
        else:
            pass

        for layeri in my_model.layers:
            if layeri.name in [j.name for j in pre_trained_model.layers]:
                temp = pre_trained_model.get_layer(layeri.name).get_weights()
                layeri.set_weights(temp)
        pre_trained_model = my_model

    x = pre_trained_model.output
    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    x = layers.Dropout(top_dropout, name='top_dropout')(x)
    predictions = layers.Dense(classes, activation='softmax')(x)
    model = Model(inputs=pre_trained_model.input, outputs=predictions)
    model.summary()
    return model

def mymodels(model_str='mobilenet', input_shape=(224, 224, 3), classes=1000):
    model = ''
    if model_str in 'VGG16':
        model = myVGG16(input_shape=input_shape, classes=classes)
    elif model_str in 'ResNet101V2':
        model = myResNetV2101(input_shape=input_shape, classes=classes)
    elif model_str in 'InceptionV3':
        model = myInceptionV3(input_shape=input_shape, classes=classes)
    elif model_str in 'DenseNet169':
        model = myDenseNet169(input_shape=input_shape, classes=classes)
    elif model_str in 'NASNetMobile':
        model = myNASNetMobile(input_shape=input_shape, classes=classes)
    elif model_str in 'MobileNetV2':
        model = myMobileNetV2(input_shape=input_shape, classes=classes)
    elif model_str in 'MobileNetV3Small':
        model = myMobileNetV3Small(input_shape=input_shape, classes=classes)
    return model

if __name__ == '__main__':
    myAlexNet()
    myVGG16()
    myResNetV2101()
    myInceptionV3()
    myXception()
    myMobileNetV2()
    myMobileNetV3Small()
    myDenseNet169()
    myDenseNet201()
    myNASNetMobile()
    myEfficientNetB0()

# 2020-11-09 guangjinzheng models
