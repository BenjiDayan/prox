# import open3d as o3d
import json
import os, sys, glob
import random
import numpy as np
# import h5py
import torch
from torch.utils import data
from tqdm import tqdm
import pickle
import cv2
import time
import copy


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


test_videoID_fps5 = ['2020-06-09-16-09-56', '2020-06-13-00-01-49', '2020-06-09-16-43-21', '2020-06-12-23-54-30']
test_videoID_fps30 = ['2020-05-22-11-08-55', '2020-06-02-16-22-41', '2020-05-22-12-05-26', '2020-06-02-14-37-21',
                      '2020-05-22-10-56-28', '2020-06-02-16-29-15', '2020-05-22-11-50-49', '2020-06-02-14-04-14',
                      '2020-06-02-15-15-53', '2020-05-22-11-55-56', '2020-06-02-15-38-15', '2020-06-02-13-47-21',
                      '2020-06-02-15-32-25', '2020-06-02-15-09-26', '2020-06-02-14-30-22', '2020-06-02-13-58-14',
                      '2020-06-02-15-02-26', '2020-06-02-14-55-57', '2020-05-22-12-00-24', '2020-06-02-15-56-19',
                      '2020-06-02-16-10-13', '2020-06-02-15-22-46']

# test_videoID_fps5 = ['2020-06-13-00-01-49']
# test_videoID_fps5 = ['2020-06-09-16-53-33']
# test_videoID_fps30 = []

test_videoID_fps5.sort()
test_videoID_fps30.sort()


class TrainLoader(data.Dataset):
    def __init__(self, split='train', h=256, w=448, read_memory=False):
        self.split = split
        self.h, self.w = h, w
        self.scale = 1080/h
        self.read_memory = read_memory

        self.train_videoDir_list = []
        self.test_videoDir_list = []

        self.img_dir_list = []
        self.depth_dir_list = []
        self.bps_feat_dir_list = []
        self.cam_int_list = []

        self.pose2d_list = []
        self.pose3d_list = []


        # save data to memory
        self.img_list = []
        self.depth_list = []
        self.bps_feat_list = []


    def get_video_dir(self, data_root='../GTA-IM'):
        # fps5
        fps5_dir = os.path.join(data_root, 'FPS-5', 'preprocessed_data')
        videoID_list_fps5 = os.listdir(fps5_dir)   # 相对路径
        train_videoID_fps5 = list(set(videoID_list_fps5) - set(test_videoID_fps5))
        train_videoID_fps5.sort()
        for videoID in train_videoID_fps5:
            self.train_videoDir_list.append(os.path.join(fps5_dir, videoID))
        for videoID in test_videoID_fps5:
            self.test_videoDir_list.append(os.path.join(fps5_dir, videoID))

        # fps30
        fps30_dir = os.path.join(data_root, 'FPS-30', 'preprocessed_data')
        videoID_list_fps30 = os.listdir(fps30_dir)
        train_videoID_fps30 = list(set(videoID_list_fps30) - set(test_videoID_fps30))
        train_videoID_fps30.sort()
        for videoID in train_videoID_fps30:
            self.train_videoDir_list.append(os.path.join(fps30_dir, videoID))
        for videoID in test_videoID_fps30:
            self.test_videoDir_list.append(os.path.join(fps30_dir, videoID))


    def load_data(self, data_root='../GTA-IM'):
        self.get_video_dir(data_root)

        if self.split == 'train':
            videoDir_list = self.train_videoDir_list    # '../GTM-IM/fps5or30/preprocessed_data/2020-...'
        elif self.split == 'test':
            videoDir_list = self.test_videoDir_list

        videoDir_list = videoDir_list[0:1]  # todo: delete

        print('[{} data loader] loading data path...'.format(self.split))

        for dir in tqdm(videoDir_list):
            info_npz = np.load(os.path.join(dir, 'info_frames.npz'))
            cur_img_dir_list = glob.glob(os.path.join(dir, 'rgb_img_input', '*.jpg'))  # absolute path
            cur_img_dir_list.sort()
            self.img_dir_list += cur_img_dir_list

            cur_depth_dir_list = []
            cur_feat_dir_list = []
            cur_pose2D_list = []
            cur_pose3D_list = []
            for img_dir in cur_img_dir_list:
                seqID = img_dir[-21:-13]  # 'seq_0xxx'
                frameID = int(img_dir[-9:-4])
                cur_depth_dir_list += glob.glob(os.path.join(dir, 'depth_inpaint_npy', seqID + '*'))

                # all bps feature map dirs of current sequence ID
                bps_feats = glob.glob(os.path.join(dir, 'bps_feature_npy', seqID + '*'))  # list, 15 frames
                bps_feats.sort()
                cur_feat_dir_list.append(bps_feats)

                frs = [int(bps_feats[i][-9:-4]) for i in range(15)]   # list of frame IDs of current sequence
                cur_pose2D_list.append(info_npz['joints_2d'][frs])  # [15, 21, 2] of original [1080, 1920] plane

                # read world coordinate of cur frame, transfrom to cam coordinate by cam ext of 4th frame in this seq
                pose3D_world = info_npz['joints_3d_world'][frs]  # [15, 21, 3]
                cam_ext_T = info_npz['world2cam_trans'][frameID]
                temp = pose3D_world.reshape(-1, 3)
                temp = np.hstack([temp, np.ones([temp.shape[0], 1])])  # [15*21, 4]
                pose3D_cam = temp.dot(cam_ext_T)  # [15*21, 4]
                cur_pose3D_list.append(pose3D_cam[:, :3].reshape(15, -1, 3))  # [15, 21, 3]

                cam_int = info_npz['intrinsics'][frameID]
                self.cam_int_list.append(cam_int)


            self.depth_dir_list += cur_depth_dir_list
            self.bps_feat_dir_list += cur_feat_dir_list  # list, each element is a list of 15 paths
            self.pose2d_list += cur_pose2D_list    # list, each element: array[15,2]
            self.pose3d_list += cur_pose3D_list

        if not (len(self.img_dir_list) == len(self.depth_dir_list) == len(self.bps_feat_dir_list) ==
                len(self.pose2d_list) == len(self.pose3d_list)):
            print('[{} data loader] data size not compatable!!!'.format(self.split))

        if self.split == 'test':
            sample = 8
            self.img_dir_list = [self.img_dir_list[sample * i] for i in range(int(len(self.img_dir_list) / sample))]
            self.depth_dir_list = [self.depth_dir_list[sample * i] for i in range(int(len(self.depth_dir_list) / sample))]
            self.bps_feat_dir_list = [self.bps_feat_dir_list[sample * i] for i in range(int(len(self.bps_feat_dir_list) / sample))]
            self.pose2d_list = [self.pose2d_list[sample * i] for i in range(int(len(self.pose2d_list) / sample))]
            self.pose3d_list = [self.pose3d_list[sample * i] for i in range(int(len(self.pose3d_list) / sample))]
            self.cam_int_list = [self.cam_int_list[sample * i] for i in range(int(len(self.cam_int_list) / sample))]

        print('[{} data loader] sequence number:'.format(self.split), len(self.img_dir_list))


        if self.read_memory:
            self.read_data_to_memory()



    def __len__(self):
        return len(self.img_dir_list)


    def read_data_to_memory(self):
        print('[{} data loader] reading data into memory...'.format(self.split))
        for i in tqdm(range(len(self.img_dir_list))):
            img = cv2.imread(self.img_dir_list[i])
            depth = np.load(self.depth_dir_list[i])
            bps_seq = []
            for bps_path in self.bps_feat_dir_list[i]:
                bps_seq.append(np.load(bps_path))
            bps_seq = np.asarray(bps_seq)  # [15, h, w]

            self.img_list.append(img)       # [n, h, w, 3]
            self.depth_list.append(depth)   # [n, h, w]
            self.bps_feat_list.append(bps_seq)  # [n, 15, h, w]


    def __getitem__(self, index):
        if self.read_memory:
            img = self.img_list[index]
            img = img[:, :, ::-1].copy()  # BGR --> RGB
            img = img.transpose((2, 0, 1))  # [H,W,C] --> [C,H,W]
            img = img / 255.0               # to [0, 1]

            depth = self.depth_list[index]
            depth[depth > 20] = 20.0
            depth = depth / 20.0  # to [0, 1]

            bps_seq = self.bps_feat_list[index]
            bps_seq[bps_seq > 10] = 10.0
            bps_seq = bps_seq / 10.0

        else:
            img = cv2.imread(self.img_dir_list[index])  # [256,448,3]
            img = img[:, :, ::-1].copy()  # BGR --> RGB
            img = img.transpose((2, 0, 1))  # [H,W,C] --> [C,H,W]
            img = img / 255.0               # to [0, 1]
            depth = np.load(self.depth_dir_list[index])  # [h, w]
            depth[depth > 20] = 20.0
            depth = depth / 20.0  # to [0, 1]

            bps_path_seq = self.bps_feat_dir_list[index]
            bps_seq = []
            for bps_path in bps_path_seq:
                bps = np.load(bps_path)
                bps_seq.append(bps)
            bps_seq = np.asarray(bps_seq)
            bps_seq[bps_seq > 10] = 10.0
            bps_seq = bps_seq / 10.0


        img_dir = self.img_dir_list[index]   # '..../GTM-IM/fps5or30/preprocessed_data/2020-.../rgb_img_input/xxx.jpg'

        pose2d_seq = self.pose2d_list[index] / self.scale
        pose2d_seq = pose2d_seq.transpose((0, 2, 1))
        pose3d_seq = self.pose3d_list[index]
        pose3d_seq = pose3d_seq.transpose((0, 2, 1))

        cam_init = self.cam_int_list[index]


        img = torch.from_numpy(img).float()  # [3, h, w]
        depth = torch.from_numpy(depth).float().unsqueeze(0)  # [1, h, w]
        bps_seq = torch.from_numpy(bps_seq).float().unsqueeze(1)  # [15, 1, 256, 448]
        pose2d_seq = torch.from_numpy(pose2d_seq).float()  # [15, 2, 21]
        pose3d_seq = torch.from_numpy(pose3d_seq).float()  # [15, 3, 21]
        cam_init = torch.from_numpy(cam_init).float()  # [3, 3]?
        return [img, depth, bps_seq, pose2d_seq, pose3d_seq, cam_init, img_dir]




if __name__ == '__main__':
    data_root = '/local/home/szhang/GTA-1M'
    dataset = TrainLoader(split='test', h=256, w=448, read_memory=False)
    dataset.load_data(data_root=data_root)

    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=2)

    min = 100000000
    for step, data in tqdm(enumerate(dataloader)):
        [_, depth, bps_seq, _, pose3d_seq, cam_int] = [item.to(device) for item in data[0:-1]]
        step += 1
        # bps_seq: [bs, 15, 1, 256, 448]
        for i in range(15):
            bps = bps_seq[:, i,] # [1, h, w]
            cnt = (bps<0.3).sum().item()
            if cnt < min:
                min = cnt
    print(min)






















