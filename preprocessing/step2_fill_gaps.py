"""
    Using gap information from step 1, fill in the gaps. Different strategies are used depending on
    the size of the gap.
"""

import i3d_config as cfg
import re
import cv2
import os.path
import json
from tqdm import tqdm

colors = {'green': (0, 255, 0), 'red': (0, 0, 255), 'yellow': (0, 255, 255)}


def skip_to_frame(cap, frame_nr):
    ret_frame = 0
    for f in range(frame_nr-1):
        cap.read()
        ret_frame += 1
    return ret_frame


def draw_detections(image, frame_detections, rect_color):
    for detection in frame_detections:
        x, y, w, h = detection['rect']
        score = detection['score']
        cv2.rectangle(image, (x, y), (x + w, y + h), color=colors[rect_color], thickness=2)
        cv2.putText(image, '%.2f' % score, (x, y), cv2.FONT_HERSHEY_PLAIN, fontScale=.6, thickness=1,
                    color=colors['yellow'])


small_gap_lim = 60

fp = open('detection_gaps.csv', 'r')
lines = fp.readlines()
fp.close()

videos_dir = cfg.dataset_dir + '/mp4'

chunks = re.split(';', lines[0][:-1])  # last char is newline.
curr_sequence_name = chunks[0].strip()
with open(curr_sequence_name) as fh:
    working_detections = json.load(fh)

lines.append("STOP;")

for line in tqdm(lines):
    chunks = re.split(';', line[:-1])  # last char is newline.
    sequence_name = chunks[0].strip()
    if not sequence_name == "STOP":
        start = int(chunks[1])
        duration = int(chunks[2])
        end = int(chunks[3])

    if sequence_name != curr_sequence_name:
        modified_sequence_name = curr_sequence_name.replace('mp4_mrcnn_bbox', 'mp4_mrcnn_bbox_nogaps')
        modified_sequence_path = os.path.split(modified_sequence_name)[0]
        if not os.path.exists(modified_sequence_path):
            os.makedirs(modified_sequence_path)

        with open(modified_sequence_name, 'w') as fh:
            json.dump(working_detections, fh)
        if sequence_name == "STOP":
            print('Done.')
            break
        with open(sequence_name, 'r') as fh:
            working_detections = json.load(fh)

        curr_sequence_name = sequence_name
        print('-'*40)

    if duration < small_gap_lim:

        # video_name = sequence_name.replace('mp4_mrcnn_bbox', 'mp4').replace('.json', '.mp4')
        # cap = cv2.VideoCapture(video_name)
        print(sequence_name, start, end, duration)

        # skip_to_frame(cap, start)
        frame_before = start - 1
        for i in range(duration+1):
            # ret, frame = cap.read()

            if i < duration:
                frame_detections = working_detections['%06d' % frame_before]
                working_detections['%06d' % (start + i)] = working_detections['%06d' % frame_before]
            else:
                frame_detections = working_detections['%06d' % end]

            # draw_detections(frame, frame_detections, 'green' if i == 0 or i == duration else 'red')
            # cv2.imshow('frame', frame)
            # cv2.waitKey(20)




