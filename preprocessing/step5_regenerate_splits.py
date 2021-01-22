"""
    Splits from previous training (LSTMs x3 net) are not valid now, because of lack of detections for some sequences
    in video.
"""
import sys
sys.path.insert(0,'/home/pau/projects/harnets/ntu-i3d')
# sys.path.insert(0,'/home/paal/deploy/ntu-i3d')

import i3d_config as cfg
from glob import glob
import os.path

split_files = glob('%s/splits/*CV*' % cfg.dataset_dir)

check_dir = cfg.crops_dir

for split_file in split_files:
    fh = open(split_file, 'r')
    lines = fh.readlines()
    fh.close()

    fd = open(os.path.split(split_file)[1], 'w')

    valid = 0
    total = 0
    for line in lines:
        sequence_name = os.path.split(os.path.splitext(line)[0])[1]
        if os.path.exists('%s/%s' % (check_dir, sequence_name)):
            valid += 1
            fd.write(line)
        total += 1

    print('Percentage of valid for split file "%s": %3.1f%% (%05d/%05d)'
          % (split_file, 100 * (valid/total), valid, total))

    fd.close()
