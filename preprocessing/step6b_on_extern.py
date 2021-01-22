"""
    Check all images are valid!
"""
import sys

import cv2
from glob import glob
from tqdm import tqdm

sequences = glob('/media/francisco/extern_wd/Datasets/ToyotaSmartHome/mp4_mrcnn_cropped/*')
print('Found %d sequences.' % len(sequences))

fh = open('image_errors_extern.log', 'w')

for sequence in tqdm(sequences):
    files = glob('%s/*' % sequence)
    for file in files:
        # print(file)
        img = cv2.imread(file)
        if img is None or img.size == 0:
            print(file)
            fh.write('%s\n' % file)

fh.close()


