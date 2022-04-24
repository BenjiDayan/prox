import os
import torch
# import pandas as pd
# from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import glob
from pathlib import Path
import re
import datetime as dt
import pickle

class proxDataset(Dataset):
    def __init__(self, root_dir, in_frames=10, pred_frames=5):
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

        

        # step = 7
        # pred = 3
        # ys = np.arange(0, n, step=step+pred)
        # in_frames = [list(range(i, i+step)) for i in ys[:-1]]
        # pred_frames = [list(range(i+step, i+step+pred)) for i in ys[:-1]]


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

        def load(fn):
            with open(fn, 'rb') as file:
                return pickle.load(file) 
        
        in_data, pred_data = map(lambda fns: [load(fn) for fn in fns],  [in_frames_fns, pred_frames_fns])
        in_skels, pred_skels = map(lambda datas: np.array([data['body_pose'] for data in datas]).reshape(-1, 21, 3), [in_data, pred_data])
        return (idx, in_skels, pred_skels)



