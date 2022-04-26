import os
import torch
# import pandas as pd
# from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.utils.data._utils.collate import default_collate
from torchvision import transforms, utils
import glob
from pathlib import Path
import re
import datetime as dt
import pickle
import smplx


def load(fn):
    try:
        with open(fn, 'rb') as file:
            return pickle.load(file) 
    except FileNotFoundError:
        return None

def nans_of_shape(shape):
    out = np.empty(shape)
    out[:] = np.nan
    return out

class proxDataset(Dataset):
    def __init__(self, root_dir, in_frames=10, pred_frames=5, output_type='joint_locations', smplx_model_path=None):
        # NB output_type='joint_locations' is deprecated in favor of preprocess_joint_locs.py (which could be ported in here tbh)
        if not output_type in ['joint_locations', 'joint_thetas', 'raw_pkls']:
            raise Exception("output_type should be one of ['joint_locations', 'joint_thetas', 'raw_pkls']")

        # we need a body model to convert beta, theta and global translation into 3D joint locations.
        if output_type == 'joint_locations':
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            body_model = smplx.create(smplx_model_path, 
                          model_type='smplx',        ## smpl, smpl+h, or smplx?
                          gender='neutral', ext='npz',  ## file format 
                          num_pca_comps=12,          ## MANO hand pose pca component
                          create_global_orient=True, 
                          create_body_pose=True,
                          create_betas=True,
                          create_left_hand_pose=True,
                          create_right_hand_pose=True,
                          create_expression=True, 
                          create_jaw_pose=True,
                          create_leye_pose=True,
                          create_reye_pose=True,
                          create_transl=True,
                          batch_size=1               ## how many bodies in a batch?
                          )
            body_model.eval()
            self.body_model = body_model

        self.output_type = output_type
        self.root_dir = Path(root_dir)

        scenes = glob.glob(str(self.root_dir / '*'))
        scenes = [Path(scene) for scene in scenes]
        scenes_dir = {scene.name: list(map(Path, glob.glob(str(scene / 'results/*')))) for scene in scenes}
        scenes_dir2 = {stem: [(str(path / '000.pkl'), path.name)  for path in paths] for stem, paths in scenes_dir.items()}
        outputs = {}
        for stem, paths in scenes_dir2.items():
            if not stem in outputs:
                outputs[stem] = {}
            for path, name in paths:
                scene, frame, tstamp = re.match(r's(\d*)_frame_(\d*)__(.*)', name).groups()
                scene, frame = int(scene), int(frame)
                tstamp = dt.datetime.strptime(tstamp, '%H.%M.%S.%f').time()
                if scene not in outputs[stem]:
                    outputs[stem][scene] = []
                outputs[stem][scene].append({'fn': path, 'frame': frame, 'tstamp': tstamp})

        sequences = []
        for stem in outputs:
            for scene, frame_dicts in outputs[stem].items():
                sequences.append([stem, sorted(frame_dicts, key=lambda x: x['frame'])])

        # ('BasementSittingBooth_00142_01',
        # [{'fn': 'D:\\prox_data\\PROXD_attempt2\\PROXD\\BasementSittingBooth_00142_01\\results\\s001_frame_00001__00.00.00.029\\000.pkl',
        # 'frame': 1,
        # 'tstamp': datetime.time(0, 0, 0, 29000)},
        # {'fn': 'D:\\prox_data\\PROXD_attempt2\\PROXD\\BasementSittingBooth_00142_01\\results\\s001_frame_00002__00.00.00.050\\000.pkl',
        # 'frame': 2, ...])
        # (around 1500 frames per sequence). Mutiple sequences will be in a similar 3D environment, e.g. 
        # [stem for stem, fns_dict in sequences.items()]
            # 'BasementSittingBooth_00142_01',
            # 'BasementSittingBooth_00145_01',
            # 'BasementSittingBooth_03452_01',
            # 'MPH11_00034_01',
            # 'MPH11_00150_01', ...
        self.sequences = sequences
        self.in_frames = in_frames
        self.pred_frames = pred_frames
        self.tot_frames = in_frames + pred_frames
        seq_lens = [len(fns_dict) for stem, fns_dict in sequences]
        self.bounds = np.array([seq_len // self.tot_frames for seq_len in seq_lens])  # e.g. 
        assert(np.all(self.bounds >= 1)), "sequence has insufficient frames for one training input"  # sanity check
        self.bounds = np.cumsum(self.bounds)


    def __len__(self):
        return self.bounds[-1]
    
    def __getitem__(self, idx):
        seq_idx = np.digitize(idx, self.bounds)
        assert seq_idx < len(self.bounds), "idx too big"
        idx_in_seq = idx - (self.bounds[seq_idx-1] if seq_idx > 0 else 0)
        start = idx_in_seq*self.tot_frames

        in_frames_dicts = self.sequences[seq_idx][1][start:start+self.in_frames:1]
        pred_frames_dicts = self.sequences[seq_idx][1][start+self.in_frames:start+self.tot_frames:1]
        in_frames_fns = [frame_dict['fn'] for frame_dict in in_frames_dicts]
        pred_frames_fns = [frame_dict['fn'] for frame_dict in pred_frames_dicts]

        
        in_data, pred_data = map(lambda fns: [load(fn) for fn in fns],  [in_frames_fns, pred_frames_fns])
        # In event of failed file read, have arrays of appropriate shape but filled with nans - these training pairs
        # will be filtered out by our defined collate_fn in batching.
        if None in in_data or None in pred_data:
            # in_skels, pred_skels = nans_of_shape((self.in_frames, 21, 3)), nans_of_shape((self.pred_frames, 21, 3))
            in_skels, pred_skels = None, None
        elif self.output_type == 'joint_thetas':  # .reshape(-1, 21, 3)
            in_skels, pred_skels = map(
                lambda datas: torch.stack([torch.Tensor(data['body_pose']) for data in datas], dim=0).reshape(-1, 21, 3),
                [in_data, pred_data])
        elif self.output_type == 'joint_locations':
            in_skels, pred_skels = map(
                lambda datas: torch.stack([self.body_model(return_joints=True, betas=torch.Tensor(data['betas']), body_pose=torch.Tensor(data['body_pose'])).joints[0] for data in datas], dim=0),
                [in_data, pred_data]
            )
            

        if self.output_type == 'raw_pkls':
            return (idx, (in_frames_fns, in_data), (pred_frames_fns, pred_data))
        if self.output_type == 'joint_thetas':
            return (idx, in_skels, pred_skels)
        elif self.output_type == 'joint_locations':
            return (idx, in_skels, pred_skels)

        return (idx, in_skels, pred_skels) if not self.verbose else (idx, (in_frames_fns, in_data), (pred_frames_fns, pred_data))



class proxDatasetJoints(proxDataset):
    def __init__(self, root_dir):
        super().__init__(root_dir, output_type='raw_pkls')  # for pkl loading.
        self.seq_lens = np.array([len(fns_dict) for stem, fns_dict in self.sequences])
        self.seq_lens = np.cumsum(self.seq_lens)
        
    def __len__(self):
        return self.seq_lens[-1]  # now each individual file is a full in and pred training point

    def __getitem__(self, idx):
        seq_idx = np.digitize(idx, self.seq_lens)
        assert seq_idx < len(self.seq_lens), "idx too big"
        idx_in_seq = idx - (self.seq_lens[seq_idx-1] if seq_idx > 0 else 0)
        start = idx_in_seq

        frame_seq_dict = self.sequences[seq_idx][1][start]
        fn = frame_seq_dict['fn']
        frame_seq_data = load(fn)
        in_joint_locations, pred_joint_locations = frame_seq_data['in_joint_locations'], frame_seq_data['pred_joint_locations']

        return (idx, in_joint_locations, pred_joint_locations)



#  doesn't exist. there's 02919, 02920, 02970 then 02950 weirdly
# try pd2.__getitem__(2098)  this is the offending sample
# FileNotFoundError: [Errno 2] No such file or directory: 'D:\\prox_data\\PROXD_attempt2\\PROXD\\MPH1Library_00145_01\\results\\s001_frame_02920__00.01.37.300\\000.pkl'

def my_collate(batch):
    # I think these are still np.arrays, will become tensors later
    batch = list(filter(
        # check that they exist and don't have nans??
        lambda triple: (triple[1] is not None) and (triple[2] is not None) and (not torch.any(torch.isnan(triple[1]))) and (not torch.any(torch.isnan(triple[2]))),
        batch
    ))
    return default_collate(batch)