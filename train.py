from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from multiprocessing import Process
from PIL import ImageFile
import tensorflow as tf
import tensorflow_addons as tfa
import datetime
import time
import csv
import os
import argparse
import models
ImageFile.LOAD_TRUNCATED_IMAGES = True

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default='Flavia', help="is Flavia, Flower, Leafsnap, Cifar or Swedish")
parser.add_argument("--models", type=str, default='EfficientNetB0', help="is EfficientNetB0~B3, VGG16\
                    ResNetV2101, InceptionV3, DenseNet169, NASNetMobile or MobileNetV3")
parser.add_argument("--epochs", type=int, default=100, help="number of epochs of training")
parser.add_argument("--lr", type=float, default=5e-5, help="Adam: learning rate")
parser.add_argument("--af", type=str, default='swish', help="is relu, swish or hswish")
parser.add_argument("--at", type=str, default='se', help="is se or eca")
parser.add_argument("--dirs", type=str, default='', help="is model data path")
parser.add_argument("--load", type=int, default=0, help="number of models")
parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
parser.add_argument("--img_size", type=int, default=224, help="size of each image dimension")
parser.add_argument("--num", type=int, default=1, help="number of running train.py")
opt = parser.parse_args()

# train model
def trainmodel():
    path = 'D:/deeplearning/datasets/imageclassification/'
    if opt.data in 'Flavia-32':
        path += 'Flavia-32'
    elif opt.data in 'Flower-102':
        path += 'Flower-102'
    elif opt.data in 'Leafsnap-184':
        path += 'Leafsnap-184'
    elif opt.data in 'Swedish-15':
        path += 'Swedish-15'
    elif opt.data in 'Cifar-100':
        path += 'Cifar-100'
    else:
        path += 'Flavia-32'
    classes = int(path.split('-')[-1].split('/')[0])
    print(opt)
    if 'EfficientNet' in opt.models:
        model = models.myEfficientNet(model_str=opt.models, attention=opt.at, activation=opt.af,
                                      input_shape=(opt.img_size, opt.img_size, 3), classes=classes)
    else:
        model = models.mymodels(model_str=opt.models, input_shape=(opt.img_size, opt.img_size, 3), classes=classes)
    METRICS = [
        'accuracy',
        tf.keras.metrics.Precision(name='Precision'),
        tf.keras.metrics.Recall(name='Recall')
    ]
    model.compile(optimizer=Adam(opt.lr), loss='categorical_crossentropy', metrics=METRICS)
    # load data
    train_datagen = ImageDataGenerator(rescale=1./255., rotation_range=40, width_shift_range=0.2,
                                       height_shift_range=0.2, shear_range=0.2, zoom_range=0.2,
                                       horizontal_flip=True)
    test_datagen = ImageDataGenerator(rescale=1.0/255.)
    train_generator = train_datagen.flow_from_directory("{}/train/".format(path),
                      batch_size=opt.batch_size, class_mode='categorical', target_size=(opt.img_size, opt.img_size))
    test_generator = test_datagen.flow_from_directory("{}/test/".format(path),
                      batch_size=opt.batch_size, class_mode='categorical', target_size=(opt.img_size, opt.img_size))
    # callback
    timenow = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    opt.dirs = 'logs/{}/{}/'.format(opt.models, timenow)
    os.makedirs('{}epoch'.format(opt.dirs))
    tensorboard_callback = TensorBoard(log_dir="{}".format(opt.dirs), histogram_freq=1)
    cp_callback = ModelCheckpoint(filepath=opt.dirs+'epoch/{epoch:04d}.h5', period=1, save_weights_only=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='accuracy', verbose=1, factor=0.2, patience=5, min_lr=1e-10)
    # load weights
    if opt.load > 0:
        model.load_weights('')
    history = model.fit(train_generator, epochs=opt.epochs, callbacks=[tensorboard_callback, cp_callback, reduce_lr])
    modelnum = history_csv(model, test_generator, history.history, pathcsv='{}/{}-{}.csv'.format(opt.dirs, opt.models, timenow))
    model.load_weights('{}epoch/{:04d}.h5'.format(opt.dirs, modelnum))
    score = model.evaluate(test_generator)
    model.save('{}{}-{:.6f}-{:.4f}.h5'.format(opt.dirs, opt.models, score[0], score[1]*100))
    print('{}'.format(score))

# save loss acc
def history_csv(model, test, history, pathcsv='plt.csv'):
    str_lossacc = ['id', 'loss', 'accuracy', 'Precision', 'Recall',
                   'test_loss', 'test_accuracy', 'test_Precision', 'test_Recall']
    epochs = len(history[str_lossacc[1]])
    modelmax, modelnum = 0, 0
    with open(pathcsv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=str_lossacc)
        writer.writeheader()
        for i in range(epochs):
            print('{}/{}'.format(i + 1, opt.epochs))
            model.load_weights("{}epoch/{:04d}.h5".format(opt.dirs, i + 1))
            score = model.evaluate(test)
            writer.writerow({str_lossacc[0]: '{}'.format(i + 1),
                             str_lossacc[1]: history[str_lossacc[1]][i], str_lossacc[2]: history[str_lossacc[2]][i],
                             str_lossacc[3]: history[str_lossacc[3]][i], str_lossacc[4]: history[str_lossacc[4]][i],
                             str_lossacc[5]: '{}'.format(score[0]), str_lossacc[6]: '{}'.format(score[1]),
                             str_lossacc[7]: '{}'.format(score[2]), str_lossacc[8]: '{}'.format(score[3])})
            if score[1] > modelmax:
                modelmax = score[1]
                modelnum = i + 1
        writer.writerow({str_lossacc[0]: opt})
    f.close()
    return modelnum

def times(x=0):
    arr_data = ['Flower', 'Leaf', 'Fruit']
    arr_at = ['eca', 'se']
    arr_af = ['hswish', 'swish', 'relu']
    num = 0
    for i in range(len(arr_data)):
        for j in range(len(arr_at)):
            for k in range(len(arr_af)):
                num = num + 1
                if num == x:
                    opt.data = arr_data[i]
                    opt.at = arr_at[j]
                    opt.af = arr_af[k]
                    print('{} {} {}'.format(opt.data, opt.at, opt.af))
                    break

if __name__ == '__main__':
    for i in range(opt.num):
        trainmodel()
    # modelx = ['EfficientNetB0', 'VGG16', 'ResNet101V2', 'InceptionV3', 'NASNetMobile',
    #           'DenseNet169', 'MobileNetV3']
    # opt.data = 'Leaf'
    # for i in modelx:
    #     opt.models = i
    #     trainmodel()
    # for i in range(opt.num):
    #     times(i+1)
    #     trainmodel()
    # for i in range(opt.num):
    #     times(i+1)
    #     if i != 0:
    #         time.sleep(60 * 2)
    #     p = Process(target=trainmodel)
    #     p.start()
    #     p.join()
    pass

# 2021-06-13 guangjinzheng tensorflow efficientnet
