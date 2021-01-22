"""
    Check all images are valid!
"""
import sys
sys.path.insert(0,'/home/pau/projects/harnets/ntu-i3d')
# sys.path.insert(0,'/home/paal/deploy/ntu-i3d')
import i3d_config as cfg

import cv2
from glob import glob
from tqdm import tqdm

sequences = glob('%s/*' % cfg.crops_dir)
print('Found %d sequences.' % len(sequences))

fh = open('image_errors.log', 'w')

for sequence in tqdm(sequences):
    files = glob('%s/*' % sequence)
    for file in files:
        # print(file)
        img = cv2.imread(file)
        if img is None or img.size == 0:
            print(file)
            fh.write('%s\n' % file)

fh.close()


