"""
    Test load_model and obtain results (per class accuracies)
"""
import numpy as np
from keras.models import Model, load_model
import i3d_config as cfg
from Smarthome_Loader import *
from tqdm import tqdm

batch_size = 4
version = 'i3d_30jun9pm'

print('Loading model ...')
# model = load_model('./results/weights_i3d_30jun9pm/epoch_36.hdf5')
model = load_model('./weights_i3d_class_weighted/epoch_38.hdf5')
print('Done.')

test_generator = DataLoader_video('%s/splits_i3d/test_CS.txt' % cfg.dataset_dir, version, batch_size=1, is_test=True)

num_tests = len(test_generator)
print('Testing %d samples.' % num_tests)

nc = test_generator.num_classes
conf_mat = np.zeros((nc+1, nc+1))

for i in tqdm(range(num_tests)):
    sample = test_generator[i]
    x, y = sample
    pred = model.predict(x)
    p = np.argmax(pred)
    t = np.argmax(y)
    conf_mat[t, p] += 1

np.savetxt("confusion_matrix_cw.csv", conf_mat, delimiter=";")
print('FINISHED.')
