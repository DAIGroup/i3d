import os

os.environ['KERAS_BACKEND'] = 'tensorflow'

from keras.utils import Sequence, to_categorical

import numpy as np
from random import sample, randint, shuffle
import glob
import cv2
import time

import i3d_config as cfg

# weights_CV = {0: 0.2312, 1: 0.0345, 2: 0.0310, 3: 0.0040, 4: 0.1290, 5: 0.0154, 6: 0.0332, 7: 0.0166, 8: 0.0131,
#               9: 0.0179, 10: 0.0334, 11: 0.1423, 12: 0.0117, 13: 0.0098, 14: 0.0298, 15: 0.0259, 16: 0.1981,
#               17: 0.0219, 18: 0.0012}
weights_CV = {0: 0.9969, 1: 0.9794, 2: 0.9771, 3: 0.8203, 4: 0.9945, 5: 0.9539, 6: 0.9786, 7: 0.9572, 8: 0.9456,
              9: 0.9603, 10: 0.9787, 11: 0.9950, 12: 0.9394, 13: 0.9272, 14: 0.9762, 15: 0.9726, 16: 0.9964,
              17: 0.9676, 18: 0.6832}

#weights_CV_2 = {0: 195.54, 1: 29.15, 2: 26.22, 3: 3.34, 4: 109.14, 5: 13.04, 6: 28.10, 7: 14.05, 8: 11.04, 9: 15.14,
#                 10: 28.27, 11: 120.33, 12: 9.92, 13: 8.26, 14: 25.23, 15: 21.93, 16: 167.61, 17: 18.55, 18: 1.00}

weights_CS = {0: 0.0000, 1: 0.0130, 2: 0.0117, 3: 0.0299, 4: 0.0085, 5: 0.0424, 6: 0.1222, 7: 0.0000, 8: 0.0152,
              9: 0.0155, 10: 0.0022, 11: 0.0536, 12: 0.0090, 13: 0.0196, 14: 0.0104, 15: 0.0063, 16: 0.0280,
              17: 0.0103, 18: 0.0000, 19: 0.0783, 20: 0.0664, 21: 0.0694, 22: 0.0848, 23: 0.0186, 24: 0.0898,
              25: 0.0000, 26: 0.0407, 27: 0.0053, 28: 0.0048, 29: 0.0156, 30: 0.0150, 31: 0.0954, 32: 0.0104,
              33: 0.0011, 34: 0.0066}


def name_to_int(name):
    integer = 0
    if name == "Cook":
        integer = 1
    elif name == "Cook.Cleandishes":
        integer = 2
    elif name == "Cook.Cleanup":
        integer = 3
    elif name == "Cook.Cut":
        integer = 4
    elif name == "Cook.Stir":
        integer = 5
    elif name == "Cook.Usestove":
        integer = 6
    elif name == "Cutbread":
        integer = 7
    elif name == "Drink":
        integer = 8
    elif name == "Drink.Frombottle":
        integer = 9
    elif name == "Drink.Fromcan":
        integer = 10
    elif name == "Drink.Fromcup":
        integer = 11
    elif name == "Drink.Fromglass":
        integer = 12
    elif name == "Eat.Attable":
        integer = 13
    elif name == "Eat.Snack":
        integer = 14
    elif name == "Enter":
        integer = 15
    elif name == "Getup":
        integer = 16
    elif name == "Laydown":
        integer = 17
    elif name == "Leave":
        integer = 18
    elif name == "Makecoffee":
        integer = 19
    elif name == "Makecoffee.Pourgrains":
        integer = 20
    elif name == "Makecoffee.Pourwater":
        integer = 21
    elif name == "Maketea.Boilwater":
        integer = 22
    elif name == "Maketea.Insertteabag":
        integer = 23
    elif name == "Pour.Frombottle":
        integer = 24
    elif name == "Pour.Fromcan":
        integer = 25
    elif name == "Pour.Fromcup":
        integer = 26
    elif name == "Pour.Fromkettle":
        integer = 27
    elif name == "Readbook":
        integer = 28
    elif name == "Sitdown":
        integer = 29
    elif name == "Takepills":
        integer = 30
    elif name == "Uselaptop":
        integer = 31
    elif name == "Usetablet":
        integer = 32
    elif name == "Usetelephone":
        integer = 33
    elif name == "Walk":
        integer = 34
    elif name == "WatchTV":
        integer = 35
    if integer == 0:
        print("ERROR: Returning zero in name_to_int (CS).")
    return integer


def name_to_int_CV(name):
    integer = 0
    if name == "Cutbread":
        integer = 1
    elif name == "Drink.Frombottle":
        integer = 2
    elif name == "Drink.Fromcan":
        integer = 3
    elif name == "Drink.Fromcup":
        integer = 4
    elif name == "Drink.Fromglass":
        integer = 5
    elif name == "Eat.Attable":
        integer = 6
    elif name == "Eat.Snack":
        integer = 7
    elif name == "Enter":
        integer = 8
    elif name == "Getup":
        integer = 9
    elif name == "Leave":
        integer = 10
    elif name == "Pour.Frombottle":
        integer = 11
    elif name == "Pour.Fromcan":
        integer = 12
    elif name == "Readbook":
        integer = 13
    elif name == "Sitdown":
        integer = 14
    elif name == "Takepills":
        integer = 15
    elif name == "Uselaptop":
        integer = 16
    elif name == "Usetablet":
        integer = 17
    elif name == "Usetelephone":
        integer = 18
    elif name == "Walk":
        integer = 19
    if integer == 0:
        print("ERROR: Returning zero in name_to_int (CV).")
    return integer


def get_video(vid_name, img_path, stride, stack_size, is_test):
    images = glob.glob(img_path + '/' + vid_name + "/*")
    images.sort()
    files = []
    if len(images) == 0:
        print('ERROR: NO IMAGES FOUND.')
        return []
    if len(images) > (stack_size * stride):
        start = (len(images) - stack_size*stride) // 2
        if not is_test:
            start = randint(0, len(images) - stack_size * stride)
        files.extend([images[i] for i in range(start, (start + stack_size * stride), stride)])
    elif len(images) < stack_size:
        files.extend(images)
        while len(files) < stack_size:
            files.extend(images)
        files = files[:stack_size]
    else:
        start = (len(images) - stack_size) // 2
        if not is_test:
            start = randint(0, len(images) - stack_size)
        files.extend([images[i] for i in range(start, (start + stack_size))])

    files.sort()

    arr = []
    for i in files:
        if os.path.isfile(i):
            arr.append(cv2.resize(cv2.imread(i), (224, 224)))
        else:
            arr.append(arr[-1])

    return arr


class DataLoader_video(Sequence):
    def __init__(self, path1, version, batch_size=4, is_test=True):
        self.batch_size = batch_size
        self.version = version
        self.path = cfg.crops_dir
        self.files = [i.strip() for i in open(path1).readlines()]
        self.stack_size = 64
        self.num_classes = 35 if cfg.experiment_type == 'cross-subject' else 19
        self.stride = 2
        self.is_test = is_test

    def __len__(self):
        return int(len(self.files) / self.batch_size)

    def __getitem__(self, idx):
        batch = self.files[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch = [os.path.splitext(i)[0] for i in batch]
        x_train = [get_video(i, self.path, self.stride, self.stack_size, self.is_test) for i in batch]
        x_train = np.array(x_train, np.float32)
        x_train /= 127.5
        x_train -= 1

        if cfg.experiment_type == 'cross-subject':
            labels = np.array([name_to_int(i.split('_')[0]) for i in batch]) - 1
        else:
            labels = np.array([name_to_int_CV(i.split('_')[0]) for i in batch]) - 1
        y_train = to_categorical(labels, num_classes=self.num_classes)

        if cfg.sample_weights:
            w = np.zeros((self.batch_size,))
            for i in range(len(labels)):
                if cfg.experiment_type == 'cross-subject':
                    w[i] = weights_CS[labels[i]]
                else:
                    w[i] = weights_CV[labels[i]]
            return x_train, y_train, w

        return x_train, y_train

    def on_epoch_end(self):
        shuffle(self.files)


# class DataLoader_video_test(Sequence):
#     def __init__(self, path1, version, batch_size=4):
#         self.batch_size = batch_size
#         self.version = version
#         self.path = cfg.crops_dir
#         self.files = [i.strip() for i in open(path1).readlines()]
#         self.stack_size = 64
#         self.num_classes = 35 if cfg.experiment_type == 'cross-subject' else 19
#         self.stride = 2
#
#     def __len__(self):
#         return int(len(self.files) / self.batch_size)
#
#     def __getitem__(self, idx):
#         batch = self.files[idx * self.batch_size: (idx + 1) * self.batch_size]
#         batch = [os.path.splitext(i)[0] for i in batch]
#         x_train = [get_video(i, self.path, self.stride, self.stack_size) for i in (batch)]
#         x_train = np.array(x_train, np.float32)
#         x_train /= 127.5
#         x_train -= 1
#
#         y_train = np.array([int(name_to_int(i.split('_')[0])) for i in batch]) - 1
#         y_train = to_categorical(y_train, num_classes=self.num_classes)
#
#         return x_train, y_train
#
#     def on_epoch_end(self):
#         shuffle(self.files)
#
#     def get_one(self, index):
#         return self.__getitem__(index)
