import cv2
import glob
import os
import pickle
from gta_utils import LIMBS, read_depthmap
from tqdm import tqdm
import numpy as np
from shutil import copyfile
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, default='../GTA-1M/FPS-30/2020-05-21-13-54-43')

args = parser.parse_args()



n_frame = len(glob.glob(os.path.join(args.data_root, '*.jpg')))  # total frame number

save_path = args.data_root + '_resize'

info = pickle.load(open(os.path.join(args.data_root, 'info_frames.pickle'), 'rb'))

h, w = 256, 448

if not os.path.exists(save_path):
    os.makedirs(save_path)

copyfile(os.path.join(args.data_root, 'info_frames.pickle'), os.path.join(save_path, 'info_frames.pickle'))
copyfile(os.path.join(args.data_root, 'realtimeinfo.gz'), os.path.join(save_path, 'realtimeinfo.gz'))
copyfile(os.path.join(args.data_root, 'info_frames.npz'), os.path.join(save_path, 'info_frames.npz'))


for i in tqdm(range(n_frame)):
    img_path = os.path.join(args.data_root, '{:05d}'.format(i) + '.jpg')
    img = cv2.imread(img_path)  # [h,w,3]
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)
    img_name = '{:05d}'.format(i) + '.jpg'
    cv2.imwrite('{}/{}'.format(save_path, img_name), img)

    depth_path = os.path.join(args.data_root, '{:05d}'.format(i) + '.png')
    depth = cv2.imread(depth_path)  # [h,w,3]
    depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)
    depth_name = '{:05d}'.format(i) + '.png'
    cv2.imwrite('{}/{}'.format(save_path, depth_name), depth)

    human_id_src_path = os.path.join(args.data_root, '{:05d}'.format(i) + '_id.png')
    human_id_dst_path = os.path.join(save_path, '{:05d}'.format(i) + '_id.png')
    copyfile(human_id_src_path, human_id_dst_path)

# for i in range(n_frame):
#     if (i+1)%6 != 0:
#         os.remove(os.path.join(data_root, '{:05d}'.format(i) + '.jpg'))
#         os.remove(os.path.join(data_root, '{:05d}'.format(i) + '.png'))
#         os.remove(os.path.join(data_root, '{:05d}'.format(i) + '_id.png'))




