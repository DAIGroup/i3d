import os
import re
os.environ['KERAS_BACKEND'] = 'tensorflow'
# os.environ['CUDA_VISIBLE_DEVICES'] = "2,3"
from keras.layers import Dense, Flatten, Dropout, Reshape
from keras import regularizers
from keras.preprocessing import image
from keras.models import Model, load_model
from keras.applications.vgg16 import preprocess_input
from keras.utils import to_categorical
from keras.optimizers import SGD
from i3d_inception import Inception_Inflated3d, conv3d_bn
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, CSVLogger, Callback
from keras.utils import Sequence, multi_gpu_model

import random
import sys
from multiprocessing import cpu_count
import numpy as np
import glob
from skimage.io import imread
import cv2
from Smarthome_Loader import *

import i3d_config as cfg

# Override configuration in case it's wrong
cfg.experiment_type = 'cross-view'

epochs = int(sys.argv[1])
model_name = sys.argv[2]
version = sys.argv[3]
weights_file = None
if len(sys.argv) > 4:
    weights_file = sys.argv[4]
num_classes = 19
batch_size = 4  # was 16
stack_size = 64


class i3d_modified:
    def __init__(self, weights='rgb_imagenet_and_kinetics'):
        self.model = Inception_Inflated3d(include_top=True, weights=weights)

    def i3d_flattened(self, num_classes=19):
        i3d = Model(inputs=self.model.input, outputs=self.model.get_layer(index=-4).output)
        x = conv3d_bn(i3d.output, num_classes, 1, 1, 1, padding='same', use_bias=True, use_activation_fn=False,
                      use_bn=False, name='Conv3d_6a_1x1')
        num_frames_remaining = int(x.shape[1])
        x = Flatten()(x)
        predictions = Dense(num_classes, activation='softmax', kernel_regularizer=regularizers.l2(0.01),
                            activity_regularizer=regularizers.l1(0.01))(x)
        new_model = Model(inputs=i3d.input, outputs=predictions)

        # for layer in ntu-i3d.layers:
        #    layer.trainable = False

        return new_model


class CustomModelCheckpoint(Callback):

    def __init__(self, model_parallel, path, epoch_from):
        super(CustomModelCheckpoint, self).__init__()

        self.save_model = model_parallel
        self.path = path
        self.nb_epoch = epoch_from

    def on_epoch_end(self, epoch, logs=None):
        self.nb_epoch += 1
        self.save_model.save(self.path + str(self.nb_epoch) + '.hdf5')


i3d = i3d_modified(weights='rgb_imagenet_and_kinetics')
model = i3d.i3d_flattened(num_classes=num_classes)
# WAS: optim = SGD(lr=0.01, momentum=0.9)
optim = SGD(lr=0.0001, momentum=0.9)
# optim = SGD(lr=0.001, momentum=0.9)
# optim = SGD(lr=0.0001, momentum=0.9)

# model = load_model("../weights3/epoch11.hdf5")
# Callbacks
# WAS: reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10)
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=10)
# filepath = '../weights3/weights.{epoch:04d}-{val_loss:.2f}.hdf5'
csvlogger = CSVLogger('i3d_' + model_name + '.csv')

if cfg.gpus >= 2:
    parallel_model = multi_gpu_model(model, gpus=cfg.gpus)
else:
    parallel_model = model

parallel_model.compile(loss='categorical_crossentropy', optimizer=optim, metrics=['accuracy'])
model.compile(loss='categorical_crossentropy', optimizer=optim, metrics=['accuracy'])

# TODO: Load model from weights
if weights_file is not None:
    print('Loading weights ...')
    model.load_weights(weights_file)
    filepath, filename = os.path.split(weights_file)
    file_no_ext, ext = os.path.splitext(filename)
    rest, last_epoch = re.split('_', file_no_ext)
    print('Done.')
    model_checkpoint = CustomModelCheckpoint(model, './weights_' + model_name + '/epoch_', int(last_epoch))
else:
    model_checkpoint = CustomModelCheckpoint(model, './weights_' + model_name + '/epoch_', 0)
# model_checkpoint = ModelCheckpoint('./weights/weights.{epoch:02d}-{val_loss:.2f}.hdf5')

train_generator = DataLoader_video('%s/splits_i3d/train_CV.txt' % cfg.dataset_dir, version,
                                   batch_size=batch_size, is_test=False)
val_generator = DataLoader_video('%s/splits_i3d/validation_CV.txt' % cfg.dataset_dir, version, batch_size=batch_size)

class_weights = {0: 195.54, 1: 29.15, 2: 26.22, 3: 3.34, 4: 109.14, 5: 13.04, 6: 28.10, 7: 14.05, 8: 11.04, 9: 15.14,
                 10: 28.27, 11: 120.33, 12: 9.92, 13: 8.26, 14: 25.23, 15: 21.93, 16: 167.61, 17: 18.55, 18: 1.00}

parallel_model.fit_generator(
    generator=train_generator,
    validation_data=val_generator,
    # class_weight=class_weights,
    epochs=epochs,
    callbacks=[csvlogger, reduce_lr, model_checkpoint],
    max_queue_size=48,
    # workers=2,
    use_multiprocessing=False,
)
