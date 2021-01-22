"""
    Script to visualise what each camera views on different actions
"""

import smarthome_classes as sc
import i3d_config as cfg
from glob import glob
import sys
import os
import re

for i in range(len(sc.labels_cv)):
    activity = sc.labels_cv[i]
    print(activity)

    route = '%s/mp4/%s*' % (cfg.dataset_dir, activity)
    videos = glob(route)

    print('  Total videos: %d' % len(videos))

    sys.stdout.write('  ')
    for c in range(7):
        cam = 'c%02d' % (c + 1)
        cx_videos = [v for v in videos if cam in v]
        if len(cx_videos) > 0:
            sys.stdout.write('%s, ' % cam)
    sys.stdout.write('\n')

# Second check, images from same activity:
print('='*60)
activity = 'Drink.Fromglass'
print(activity)

route = '%s/mp4/%s*' % (cfg.dataset_dir, activity)
videos = glob(route)

dico = {}
print('Total videos: %d' % len(videos))

for p in range(25):
    person = 'p%02d' % p
    px_videos = [v for v in videos if person in v]

p25_r00_videos = [v for v in videos if 'r00' in v and activity in v and 'p25' in v]
p25_v11_videos = [v for v in videos if 'v11' in v and activity in v and 'p25' in v]

print('p25_r00_videos')
for v in p25_r00_videos:
    print(v)

print('p25_v11_videos')
for v in p25_v11_videos:
    print(v)


