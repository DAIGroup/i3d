"""
    Steps:
    1) Use a human detector to get bounding boxes to trim the videos. Save in text format (json).

    json: {"frame": [{"score":0.85, "rect":[x, y, w, h]}, {"score":0.85, "rect":[x, y, w, h]}]}

    2) This script uses the result of (1) to produce the images.
"""
from glob import glob
import json
import i3d_config as cfg
import os.path
import sys
import matplotlib.pyplot as plt
import numpy as np

padding = 2


def is_at_image_border(rect, image_shape):
    H, W = image_shape
    x, y, w, h = rect
    p = padding
    is_at_border = False
    if (x <= p) or (y <= p) or ((x+w) >= (W-p)) or ((y+h) >= (H-p)):
        is_at_border = True

    return is_at_border


bbox_files_dir = cfg.bbox_dir
bbox_files = glob('%s/*.json' % bbox_files_dir)

vbars = np.zeros((101,))
gap_values = []
max_gap_values = []
csv_lines = []
for bbox_file in bbox_files:
    fh = open(bbox_file, 'r')
    bbox_dict = json.load(fh)
    fh.close()

    good = 0
    total = 0
    gaps = []
    curr_gap = 0
    last_seen_rect = [0,0,0,0]
    num_frames = len(bbox_dict)
    for frame in bbox_dict:
        rects = bbox_dict[frame]
        if len(rects) > 0:
            if curr_gap > 0:
                end = int(frame)
                start = end - curr_gap
                at_border = is_at_image_border(last_seen_rect, (480, 640))
                if not start == 0 and not end == num_frames-1 and not at_border:
                    line = '%s; %d; %d; %d\n' % (bbox_file, start, curr_gap, end)
                    csv_lines.append(line)
                    gaps.append(curr_gap)
            curr_gap = 0
            rects = sorted(rects, key=lambda k: k['score'], reverse=True)
            rect = rects[0]
            last_seen_rect = rect['rect']
            good += 1
        else:
            curr_gap += 1
        total += 1

    pct = int(100*(good/total))
    vbars[pct] += 1
    sys.stdout.write('%40s: Done. (%3d%%)\n' % (os.path.split(bbox_file)[1], pct))
    if len(gaps) > 0:
        max_gap_values.append(max(gaps))
        gap_values.extend(gaps)

fp = open('detection_gaps.csv', 'w')
fp.writelines(csv_lines)
fp.close()

for i in range(100,0,-1):
    vbars[i-1] += vbars[i]

# Figure showing pct. of videos vs pct. of predictions
plt.figure()
plt.plot(range(101), 100 * (vbars/len(bbox_files)))
plt.xlabel('frames with predictions (%)')
plt.ylabel('percentage of videos (%)')
plt.axis([-1,101,-1,101])

gap_histo = np.zeros((max(gap_values)+1,))
for gv in gap_values:
    gap_histo[gv] += 1

print('Number of videos: %5d' % len(bbox_files))
print('Identified gaps: %5d' % len(gap_values))
print('Instances <=30: %5d' % sum(gap_histo[:31]))
print('Instances  >30: %5d' % sum(gap_histo[31:]))
print('Instances  >60: %5d' % sum(gap_histo[61:]))
print('Instances >100: %5d' % sum(gap_histo[101:]))

gap_histo /= len(gap_values)

print('Percentage in 1-3: %3.3f' % (100 * sum(gap_histo[0:4])))
print('Percentage in 1-20: %3.3f' % (100 * sum(gap_histo[0:21])))
print('Percentage in 1-30: %3.3f' % (100 * sum(gap_histo[0:31])))
print('Percentage in 1-60: %3.3f' % (100 * sum(gap_histo[0:61])))

cum_gap_histo = np.zeros_like(gap_histo)
for i in range(len(gap_histo)):
    cum_gap_histo[i] = sum(gap_histo[:i+1])

# Figure showing: Histogram of gap size (bars) and cumulative size (coverage).
lim = 100
plt.figure()
plt.bar(range(lim), gap_histo[:lim])
plt.plot(cum_gap_histo[:lim], '*-', color='orangered')
plt.ylabel('occurrence (%)')
plt.xlabel('gap size (log)')
plt.show()

#




