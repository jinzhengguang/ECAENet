from tensorflow.keras.preprocessing.image import ImageDataGenerator
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from PIL import Image as ImagePIL
from keras_flops import get_flops
import models
import os
import csv
import time
import tensorflow as tf
import numpy as np
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# EfficientNet B0 Tesgt data
class Testefn(object):
    def __init__(self, path='D:/deeplearning/datasets/imageclassification', dataset='Flower', pathmodel='',
                 modelx='EfficientNetB0', attention='se', activation='swish', batch_size=32, input_shape=(224, 224, 3)):
        self.datasets = os.listdir(path)
        self.dataset = [dataseti for dataseti in self.datasets if dataset in dataseti][0]
        self.path = '{}/{}/test/'.format(path, self.dataset)
        self.classes = int(self.path.split('-')[-1].split('/')[0])
        self.path_model = pathmodel
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.attention = attention
        self.activation = activation
        self.modelx = modelx
        self.model = models.myEfficientNet(attention=self.attention, activation=self.activation,
                                           input_shape=self.input_shape, classes=self.classes)
        if self.modelx != 'EfficientNetB0':
            self.model = models.mymodels(model_str=self.modelx, input_shape=self.input_shape, classes=self.classes)
        self.model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(name='Precision'), tf.keras.metrics.Recall(name='Recall')])
        if self.path_model != '':
            self.model.load_weights(self.path_model)
        self.test_data = ImageDataGenerator(rescale=1.0 / 255.).flow_from_directory(
            self.path, batch_size=self.batch_size, class_mode='categorical',
            target_size=(self.input_shape[0], self.input_shape[1]))

    # Top1 Acc
    def top1acc(self):
        score = self.model.evaluate(self.test_data)
        top1 = 100.0 * score[1]
        print('Top1 Accuracy: {:.3f}%'.format(top1))
        return top1

    # Top5 Acc
    def top5acc(self):
        tf1 = 0
        sum_times = 0
        total = len(self.test_data.filepaths)
        for i in range(total):
            img_path = self.test_data.filepaths[i]
            img = tf.keras.preprocessing.image.load_img(img_path, target_size=(self.input_shape[0], self.input_shape[1]))
            x = tf.keras.preprocessing.image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = x / 255.0
            ts = time.time()
            preds = self.model.predict(x)[0]
            tn = time.time()
            sum_times += (tn - ts)
            preclass = np.asarray(preds, np.float)
            preflag = True
            for j in range(5):
                num = int(np.argmax(preclass))
                preclass[num] = -1
                # print('{} = {}'.format(j + 1, num))
                if num == self.test_data.labels[i]:
                    tf1 = tf1 + 1
                    preflag = False
                    break
            if preflag:
                print(img_path)
                print('{}'.format(np.argmax(preds)))
        print('true:{}  total:{}'.format(tf, total))
        top5 = 100.0 * tf1 / total
        print('Top5 Accuracy: {:.3f}%'.format(top5))
        latency = total / sum_times
        return top5, latency

    # Calculae FLOPs
    def flops(self):
        flops = get_flops(self.model, batch_size=1)
        g = flops / 2.0 / 10 ** 9
        print('{} {} {} {:.3f}G'.format(self.dataset, self.activation, self.attention, g))
        return g

    def png300(self):
        pic_path = 'pic'
        if not os.path.exists(pic_path):
            print('pic folder is empty...')
            return
        pic_dir = os.listdir(pic_path)
        for diri in pic_dir:
            if '300dpi' not in diri:
                img = ImagePIL.open('{}/{}'.format(pic_path, diri))
                img.save(r'{}/{}_300dpi.png'.format(pic_path, diri.split('.')[0]), dpi=(300, 300))
        print('a')


    def all(self):
        top1 = self.top1acc()
        top5, latency = self.top5acc()
        flop = self.flops()
        self.model.summary()
        print('{} {} {}'.format(self.dataset, self.activation, self.attention))
        print('Top1: {:.3f}% Top5: {:.3f}% FLOPs: {:.3f}G FLOPs: {:.3f}FPS'.format(top1, top5, flop, latency))

    # Pred
    def ypred(self):
        a = self.test_data
        y_true = self.test_data.labels
        y_pred = np.zeros(len(y_true), dtype=int)
        for i in range(len(y_true)):
            img = tf.keras.preprocessing.image.load_img(self.test_data.filepaths[i],
                                                        target_size=(self.input_shape[0], self.input_shape[1]))
            x = tf.keras.preprocessing.image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = x / 255.0
            preds = self.model.predict(x)[0]
            y_pred[i] = int(np.argmax(preds))
        return y_true, y_pred

    def acc(self):
        y_true, y_pred = self.ypred()
        m = tf.keras.metrics.Accuracy()
        m.update_state(y_true=y_true, y_pred=y_pred)
        a = m.result().numpy()
        # a = metrics.confusion_matrix(y_true, y_predict)
        # b = metrics.accuracy_score(y_true, y_predict)
        # c = metrics.precision_score(y_true, y_predict, average='samples')
        # d = metrics.recall_score(y_true, y_predict, average='samples')
        # e = metrics.f1_score(y_true, y_predict, average='samples')

    # pre
    def plot_confusion_matrix(self):
        y_true, y_pred = self.ypred()
        cm = metrics.confusion_matrix(y_true, y_pred)
        print(cm)
        classes = self.test_data.class_indices

        # plt.figure(figsize=(12, 8), dpi=100)
        plt.figure(dpi=300)
        np.set_printoptions(precision=2)

        # 在混淆矩阵中每格的概率值
        ind_array = np.arange(len(classes))
        x, y = np.meshgrid(ind_array, ind_array)
        for x_val, y_val in zip(x.flatten(), y.flatten()):
            c = cm[y_val][x_val]
            if c > 0.001:
                plt.text(x_val, y_val, "%d" % (c,), color='red', fontsize=12, va='center', ha='center')

        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.binary)
        plt.title('title')
        plt.colorbar()
        xlocations = np.array(range(len(classes)))
        xy = list(range(len(classes)))
        plt.xticks(xlocations, xy, rotation=90)
        plt.yticks(xlocations, xy)
        plt.ylabel('Actual label')
        plt.xlabel('Predict label')

        # offset the tick
        tick_marks = np.array(range(len(classes))) + 0.5
        plt.gca().set_xticks(tick_marks, minor=True)
        plt.gca().set_yticks(tick_marks, minor=True)
        plt.gca().xaxis.set_ticks_position('none')
        plt.gca().yaxis.set_ticks_position('none')
        plt.grid(True, which='minor', linestyle='-')
        plt.gcf().subplots_adjust(bottom=0.15)

        # show confusion matrix
        plt.savefig('confusion_matrix.png', format='png')
        # plt.show()
        return 0

    # save acc
    def acccsv(self):
        pathmodel = 'logs/EfficientNetB0/20210516-192026/epoch'
        pathcsv = '{}/0000.csv'.format(pathmodel)
        str_lossacc = ['id', 'test_loss', 'test_accuracy', 'test_Precision', 'test_Recall']
        models = os.listdir(pathmodel)
        with open(pathcsv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=str_lossacc)
            writer.writeheader()
            for i in models:
                print('{}/{}'.format(i.split('.')[0], len(models)))
                if '.h5' in i:
                    self.model.load_weights('{}/{}'.format(pathmodel, i))
                    score = self.model.evaluate(self.test_data)
                    writer.writerow({str_lossacc[0]: '{}'.format(i.split('.')[0]),
                                     str_lossacc[1]: '{}'.format(score[0]), str_lossacc[2]: '{}'.format(score[1]),
                                     str_lossacc[3]: '{}'.format(score[2]), str_lossacc[4]: '{}'.format(score[3])})
        f.close()

if __name__ == '__main__':
    modelx = ['EfficientNetB0', 'VGG16', 'ResNet101V2', 'InceptionV3', 'DenseNet169', 'MobileNetV3', 'NASNetMobile']
    arr_data = ['Swedish', 'Flavia', 'Flower', 'Leafsnap', 'Cifar']
    arr_at = ['eca', 'se']
    test = Testefn(modelx=modelx[0], dataset=arr_data[4], attention=arr_at[0],
                   pathmodel='EfficientNetB0-0.040671-99.7455.h5')
    # test.all()
    # test.png300()
    # test.plot_confusion_matrix()
    # test.flops()
    # test.acccsv()
    # test.acc()

# 2021-06-22 guangjinzheng
