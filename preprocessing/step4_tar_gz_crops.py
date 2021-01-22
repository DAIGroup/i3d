import sys
sys.path.insert(0,'/home/pau/projects/harnets/ntu-i3d')

import tarfile
from glob import glob
import i3d_config as cfg
import os.path
from tqdm import tqdm


print('STEP 4: Compress video folders for server upload.\n')

if len(sys.argv) == 3:
    n_splits = int(sys.argv[1])
    split = int(sys.argv[2])
else:
    n_splits = 1
    split = 0


video_folders = sorted(glob('%s/*' % cfg.crops_dir))
video_folders = [v for v in video_folders if not '.tar.gz' in v]
print('Found %d video folders.' % len(video_folders))

num_files_split = len(video_folders) // n_splits

first = split * num_files_split
last = split * num_files_split + num_files_split

if split == n_splits - 1:
    last += len(video_folders) % n_splits

print(first, last)
input("Press Enter to continue...")

split_video_folders = video_folders[first:last]

for video_folder in tqdm(split_video_folders):
    # print(video_folder)
    tarname = "%s.tar.gz" % video_folder
    tarname.replace(os.path.split(video_folder)[0], '/media/pau/Data/mp4_mrcnn_cropped')
    if not os.path.exists(tarname):
        tar = tarfile.open(tarname, "w:gz")
        tar.add(video_folder, arcname=tarname)
        tar.close()
        print('Done.')
    else:
        print('Skipped.')



