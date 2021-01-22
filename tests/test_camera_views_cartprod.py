import i3d_config as cfg
from glob import glob
import re
import os

print('='*60)

route = '%s/mp4/*' % (cfg.dataset_dir)
videos = glob(route)

# videos = [v for v in videos if 'p02' in v]
a_set = set()
v_set = set()
r_set = set()
c_set = set()
p_set = set()
for v in videos:
    fpath, fname = os.path.split(v)
    chunks = re.split('_', fname)
    axx = chunks[0]
    vxx = chunks[3]
    rxx = chunks[2]
    pxx = chunks[1]
    cxx = chunks[4][:3]
    v_set.add(vxx)
    r_set.add(rxx)
    c_set.add(cxx)
    a_set.add(axx)
    p_set.add(pxx)
print(v_set)
print(r_set)
print(c_set)
print(a_set)
print(p_set)

all_max_cams = []
max_cams = []
histo = [0]*8
a_set = sorted(a_set)
p_set = sorted(p_set)
v_set = sorted(v_set)
r_set = sorted(r_set)
fh = open('../results/existing_videos.txt', 'w')
for a in a_set:
    print(a)
    fh.write('%s\n' % a)
    for p in p_set:
        print('    %s' % p)
        fh.write('    %s\n' % p)
        for v in v_set:
            for r in r_set:
                vid_cams = [vd for vd in videos if a in vd and p in vd and v in vd and r in vd]
                if not len(vid_cams) == 0:
                    str_vidcams = "        %s_%s_%s_%s: %d" % (a, p, r, v, len(vid_cams))
                    print(str_vidcams)
                    fh.write('%s\n' % str_vidcams)
                    histo[len(vid_cams)] += 1
                    if len(vid_cams) > len(max_cams):
                        max_cams = vid_cams
                    if len(vid_cams) >= 3:
                        all_max_cams.append(max_cams)
    fh.flush()

fh.close()
print(max_cams)
print(all_max_cams)
print(histo)