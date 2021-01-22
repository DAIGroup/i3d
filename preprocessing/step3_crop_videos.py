"""
    Crop images from videos.
"""
import sys
from glob import glob

sys.path.insert(0,'/home/pau/projects/harnets/ntu-i3d')

import i3d_config as cfg
import os.path
import json
import cv2
import numpy as np
from tqdm import tqdm

debug = True

def skip_to_frame(cap, frame_nr):
    ret_frame = 0
    for f in range(frame_nr - 1):
        cap.read()
        ret_frame += 1
    return ret_frame


def get_video_detections():
    """
    This method will return a list of videos with the 'best' detections available.
    That is, the JSONs with the most complete (less gaps) detections, if available.
    :return: A set of json files, with detections, coming from either directory (original or '_nogaps')
    """
    all_files = glob('%s/*' % cfg.bbox_dir)
    nogaps_files = glob('%s_nogaps/*' % cfg.bbox_dir)

    nogaps_filenames = [os.path.split(f)[1] for f in nogaps_files]

    selected_files = []
    original = 0
    nogaps = 0
    for file in all_files:
        path, filename = os.path.split(file)
        if filename in nogaps_filenames:
            idx = nogaps_filenames.index(filename)
            selected_files.append(nogaps_files[idx])
            nogaps += 1
        else:
            selected_files.append(file)
            original += 1

    print(':: Source of detections ::')
    print('Original: ', original)
    print('No-gaps : ', nogaps)
    print('Total   : ', (original+nogaps))
    return sorted(selected_files)


def all_crops_exist(video_name, detections):
    dirname = '%s/%s' % (cfg.crops_dir, os.path.splitext(os.path.split(video_name)[1])[0])
    if not os.path.exists(dirname):
        return False

    crop_files = glob('%s/*.jpg' % dirname)

    num_detections = 0
    for detection in detections:
        if len(detections[detection]) > 0:
            num_detections += 1

    if len(crop_files) >= num_detections:
        return True
    else:
        return False


def process_one_video(cap, detections):
    prev_frame = -1
    for frame_num in tqdm(detections):
        # For non-continuous annotations, jump/skip to frame.
        if int(frame_num) != (prev_frame + 1):
            # cap.release()
            # cap = cv2.VideoCapture(video_name)
            diff_frames = int(frame_num) - prev_frame
            new_frame = skip_to_frame(cap, diff_frames)
        # Otherwise, simply read next frame.
        ret, frame = cap.read()
        frame_data = detections[frame_num]
        canvas = frame.copy()

        if len(frame_data) > 0:
            rect_data = frame_data[0]
            x, y, w, h = rect_data['rect']
            cv2.rectangle(canvas, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=1)

            score = rect_data['score']
            cx, cy = int(round(x + (w / 2))), int(round(y + (h / 2)))
            # 1) Identify longer rect. side.
            side = max(w, h)
            # 2) Increase by fixed pct (%), calculate square coords.
            side *= (1 + 0.2)
            side = max(256, side)
            # 3) Replicate image contents (mirror) -- if coords outside image.
            s = int(round(side / 2))
            side = int(round(side))
            rows, cols, ch = frame.shape

            cropfield = np.zeros((rows + (2 * side), cols + (2 * side), ch), dtype=frame.dtype)
            cropfield[side:side + rows, side:side + cols] = frame
            cropfield[0:side, :] = cropfield[side + 1, :]
            cropfield[side + rows:, :] = cropfield[side + rows - 1, :]
            cropfield[:, side + cols:] = cropfield[:, side + cols - 1].reshape(rows + (2 * side), 1, ch)
            cropfield[:, 0:side] = cropfield[:, side + 1].reshape(rows + (2 * side), 1, ch)

            cx, cy = cx + side, cy + side
            # cv2.rectangle(cropfield, (cx-s, cy-s), (cx+s, cy+s), color=(0,255,255), thickness= 1)
            # 4) Perform cropping
            crop = cropfield[cy - s:cy + s, cx - s:cx + s]
            crop = cv2.resize(crop, (256, 256))

            dirname = '%s/%s' % (cfg.crops_dir, os.path.splitext(os.path.split(video_name)[1])[0])
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            cropname = '%s/%s.jpg' % (dirname, frame_num)
            cv2.imwrite(cropname, crop)

        prev_frame = int(frame_num)

        cv2.imshow('image', canvas)
        cv2.imshow('cropfield', cropfield)
        cv2.imshow('crop', crop)
        cv2.waitKey(20)

def process_one_video_fullcrop(cap, detections):
    xmin, ymin, xmax, ymax = 1000, 1000, 0, 0
    rects = []  # just for figure-generating purposes
    frames = []  # same.
    for frame_num in tqdm(detections):
        frame_data = detections[frame_num]
        if len(frame_data) > 0:
            rect_data = frame_data[0]
            x, y, w, h = rect_data['rect']
            rects.append(rect_data['rect'])
            if x < xmin:
                xmin = x
            if y < ymin:
                ymin = y
            if x+w > xmax:
                xmax = x+w
            if y+h > ymax:
                ymax = y+h

    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        bside = max(ymax-ymin, xmax-xmin, 256)
        cx, cy = xmin+((xmax-xmin)//2), ymin+((ymax-ymin)//2)
        nymin, nymax, nxmin, nxmax = cy-bside//2, cy+bside//2, cx-bside//2, cx+bside//2
        # print(nymin, nymax, nxmin, nxmax)
        rows, cols, ch = frame.shape
        cropfield = np.ones((rows + (2 * bside), cols + (2 * bside), ch), dtype=frame.dtype)*128

        fcopy = frame.copy()
        for rect in rects:
            x, y, w, h = rect
            cx, cy = x+w//2, y+h//2
            cv2.rectangle(fcopy, (cx-1,cy-1), (cx+1,cy+1), (0,255,0), thickness=-1)

        cropfield[bside:bside + rows, bside:bside + cols] = frame
        frames.append(cropfield.copy())
        cropfield[bside:bside + rows, bside:bside + cols] = fcopy
        fymin, fymax, fxmin, fxmax = nymin+bside, nymax+bside, nxmin+bside, nxmax+bside
        crop = cv2.resize(cropfield[fymin:fymax, fxmin:fxmax].copy(), dsize=(256,256))
        # cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), color=(0, 255, 255), thickness=1)
        # cv2.rectangle(frame, (nxmin, nymin), (nxmax, nymax), color=(0, 255, 0), thickness=1)
        # cv2.imshow('frame', frame)
        # cv2.imshow('crop', crop)
        # cv2.imshow('cropfield', cropfield)
        # cv2.waitKey(2)


        dirname = '%s/%s' % (cfg.crops_dir, os.path.splitext(os.path.split(video_name)[1])[0])
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        cropname = '%s/%06d.jpg' % (dirname, frame_num)
        if not debug:
            cv2.imwrite(cropname, crop)
        frame_num += 1

    if debug and not ret:
        #cv2.rectangle(frame, (nxmin, nymin), (nxmax, nymax), color=(0, 255, 255), thickness=2)
        # cv2.rectangle(frame, (xmin,ymin), (xmax, ymax), (255,0,255), thickness=2)
        r = 0
        for rect in rects:
            if r % 5 == 0:
                x, y, w, h = rect
                cx, cy = x+w//2, y+h//2
                s = max(w,h)//2
                tlx, tly = bside+cx-s, bside+cy-s
                brx, bry = bside+cx+s, bside+cy+s
                rth_crop = frames[r][tly:bry,tlx:brx]
                cv2.rectangle(rth_crop, (s-1,s-1), (s+1,s+1), (0,255,0), thickness=-1)
                cv2.imshow('r%d' % r, rth_crop)
                cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), thickness=1)
            r+=1
        # cv2.imshow('image', frame)
        # cv2.imshow('cropfield', cropfield)
        cv2.imshow('crop', crop)
        cv2.waitKey(0)


if __name__ == '__main__':
    if len(sys.argv) == 3:
        n_splits = int(sys.argv[1])
        split = int(sys.argv[2])
    else:
        n_splits = 1
        split = 0

    all_detection_files = get_video_detections()

    num_files_split = len(all_detection_files) // n_splits

    first = split*num_files_split
    last = split*num_files_split + num_files_split

    if split == n_splits - 1:
        last += len(all_detection_files) % n_splits

    print(first,last)
    # input("Press Enter to continue...")

    detection_files = all_detection_files[first:last]

    if debug:
        detection_files = [df for df in detection_files if 'Walk' in df]
        # detection_files = [detection_files[1]]

    for d, detection_file in enumerate(detection_files):
        # if 'Enter' in detection_file:
        with open(detection_file, 'r') as fh:
            pct = 100 * (d/len(detection_files))
            print('Processing video %d of %d [%3.1f%%] ...' % (d, len(detection_files), pct))
            video_name = detection_file.replace('mp4_mrcnn_bbox', 'mp4').replace('.json', '.mp4')
            if not os.path.exists(video_name):
                video_name = detection_file.replace('mp4_mrcnn_bbox_nogaps', 'mp4').replace('.json', '.mp4')

            detections = json.load(fh)

            if not debug:
                if all_crops_exist(video_name, detections):
                    print('Skipped.')
                    continue

            cap = cv2.VideoCapture(video_name)

            #  This call can be replaced according to the
            #  modality to use to generate the crops.
            process_one_video_fullcrop(cap, detections)





