import argparse
import os
import pickle
import sys
import cv2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm       # some fancy visual effect
import glob
import copy
from projection_utils import Projection
from data_parser import *

# _____________ parse args _____________
parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, default='../quantitative/recordings')
parser.add_argument('--sequence_id', type=str, default='vicon_03301_01')
parser.add_argument('--save_root', type=str, default='../proximity_map')

args = parser.parse_args()

if __name__ == '__main__':
    # _____________ init _____________
    projection = Projection('../quantitative/calibration')
    depth_scale = 1e3       # TODO: determine depth_scale
    point_cloud = o3d.geometry.PointCloud()
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=448, height=256, visible=True)

    # _____________ make output directories _____________
    save_proximity_img_path = os.path.join(args.save_root, args.sequence_id, 'proximity_img')
    save_proximity_npy_path = os.path.join(args.save_root, args.sequence_id, 'proximity_npy')
    save_depth_img_path = os.path.join(args.save_root, args.sequence_id, 'depth_img')
    save_depth_npy_path = os.path.join(args.save_root, args.sequence_id, 'depth_npy')
    save_rgb_path = os.path.join(args.save_root, args.sequence_id, 'rgb_img')

    # _____________ collect paths of all frames _____________
    rgb_list = glob.glob(os.path.join(args.data_root, args.sequence_id, 'Color/*.jpg'))
    rgb_list.sort()
    n_frame = int(rgb_list[-1][-33:-30])  # total frame number

    # _____________ make 3d point cloud _____________
    for curr_frame in rgb_list:
        mask = cv2.imread(os.path.join(args.data_root, args.sequence_id, 'BodyIndexColor/' + curr_frame[-34:-4] + '.png'), cv2.IMREAD_GRAYSCALE)
        depth_im = cv2.imread(os.path.join(args.data_root, args.sequence_id, 'Depth/' + curr_frame[-34:-4] + '.png'), flags=-1).astype(float)
        depth_im = depth_im / 8.
        depth_im = depth_im * depth_scale
        scan_dict = projection.create_scan(mask, depth_im, mask_on_color = True)        # points & colors
        point_cloud.points.extend(scan_dict.get('points'))
        point_cloud.colors.extend(scan_dict.get('colors'))

    o3d.visualization.draw_geometries([point_cloud],
                                zoom=0.3412,
                                front=[0.4257, -0.2125, -0.8795],
                                lookat=[2.6172, 2.0475, 1.532],
                                up=[-0.0694, -0.9768, 0.2024])

    # _____________ naive implementation for just one frame _____________
    curr_frame = n_frame // 2

    # _____________ read keypoints & skeleton _____________
    keyp_tuple = read_keypoints('../quantitative/keypoints/' + curr_frame[-34:-4], use_hands=False, use_face=False, use_face_contour=False)
    keypoints = np.stack(keyp_tuple.keypoints)
    

    # compute distance of point cloud to human skeletan

    # create proximity map based on the distance

    pass