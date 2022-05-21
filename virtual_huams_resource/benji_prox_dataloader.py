import os
import cv2
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
from projection_utils import Projection

from utils import normalized_joint_locations


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



class DatasetBase(Dataset):
    def __init__(self, root_dir='./', in_frames=10, pred_frames=5, search_prefix='results', extra_prefix='000.pkl'):
        self.root_dir = Path(root_dir)

        scenes = glob.glob(str(self.root_dir / '*'))
        scenes = [Path(scene) for scene in scenes]
        scenes_dir = {scene.name: list(map(Path, glob.glob(str(scene / search_prefix / '*')))) for scene in scenes}
        scenes_dir = {stem: [(str(path / (extra_prefix if extra_prefix else '')), path.name) for path in paths] \
                      for stem, paths in scenes_dir.items()}
        outputs = {}
        for stem, paths in scenes_dir.items():
            if not stem in outputs:
                outputs[stem] = {}
            for path, name in paths:
                # last group is an optionally matching thing to catch e.g. .jpg, .png if it's not a folder
                scene, frame, tstamp = re.match(r's(\d*)_frame_(\d*)__((?:\d*[.]){3}\d*)(?:[.].*)?', name).groups()
                scene, frame = int(scene), int(frame)
                try:
                    tstamp = dt.datetime.strptime(tstamp, '%H.%M.%S.%f').time()
                except Exception as e:
                    print(tstamp)
                    print(name)
                    print(e.message, e.args)
                    raise e
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
        # for debugging
        self.scenes_dir = scenes_dir
        self.outputs = outputs

        self.in_frames = in_frames
        self.pred_frames = pred_frames
        self.tot_frames = in_frames + pred_frames

        seq_lens = [len(fns_dict) for stem, fns_dict in sequences]
        self.bounds = np.array([seq_len // self.tot_frames for seq_len in seq_lens])  # e.g.
        assert (np.all(self.bounds >= 1)), "sequence has insufficient frames for one training input"  # sanity check
        self.bounds = np.cumsum(self.bounds)

    def __len__(self):
        seq_lens = [len(fns_dict) for stem, fns_dict in self.sequences]
        self.bounds = np.array([seq_len // self.tot_frames for seq_len in seq_lens])  # e.g.
        assert (np.all(self.bounds >= 1)), "sequence has insufficient frames for one training input"  # sanity check
        self.bounds = np.cumsum(self.bounds)
        return self.bounds[-1]

    def __getitem__(self, idx):
        seq_idx = np.digitize(idx, self.bounds)
        assert seq_idx < len(self.bounds), "idx too big"
        idx_in_seq = idx - (self.bounds[seq_idx - 1] if seq_idx > 0 else 0)
        start = idx_in_seq * self.tot_frames

        in_frames_dicts = self.sequences[seq_idx][1][start:start + self.in_frames:1]
        pred_frames_dicts = self.sequences[seq_idx][1][start + self.in_frames:start + self.tot_frames:1]
        in_frames_fns = [frame_dict['fn'] for frame_dict in in_frames_dicts]
        pred_frames_fns = [frame_dict['fn'] for frame_dict in pred_frames_dicts]

        return in_frames_dicts, in_frames_fns, pred_frames_dicts, pred_frames_fns

    @staticmethod
    def fill(fn_dicts, frame_max=0):
        """
        NB frames start at 1
        [1, 2, 4, 5] with frame_max 7 would go [1, 2, None, 4, 5, None, None]
        """
        new_fn_dicts = []
        frame = 1
        for fn_dict in fn_dicts:
            fn_frame = fn_dict['frame']
            while fn_frame > frame:
                new_fn_dicts.append({'fn': None, 'tstamp': None, 'frame': frame})
                frame += 1
            new_fn_dicts.append(fn_dict)
            frame += 1
        
        while frame_max >= frame:
            new_fn_dicts.append({'fn': None, 'tstamp': None, 'frame': frame})
            frame += 1

        return new_fn_dicts

    def align(self, other_dataset):
        """
        one scene in self.sequences might have frames [1, 2, 4, 5], while in other.sequences [1, 4, 5, 6, 7]. 
        Output [1, 2, None, 4, 5, None, None] and [1, None, None, 4, 5, 6, 7] - using fill on that scene's fns_dicts.
        To do this we need to do self.align(other_dataset) and other_dataset.align(self)
        """

        other_scenes = [scene[0] for scene in other_dataset.sequences]
        other_maxes = [scene[1][-1]['frame'] for scene in other_dataset.sequences]
        my_shared_scenes = [scene for scene in self.sequences if scene[0] in other_scenes]

        new_scenes = []
        for scene in my_shared_scenes:
            frame_max = max(scene[1][-1]['frame'], other_maxes[other_scenes.index(scene[0])])
            new_fn_dicts = self.fill(scene[1], frame_max=frame_max)
            new_scenes.append([scene[0], new_fn_dicts])

        self.sequences = new_scenes


class proxDatasetSkeleton(DatasetBase):
    def __init__(self, output_type='joint_locations', smplx_model_path=None, **kwargs):
        super().__init__(**kwargs)

        # NB output_type='joint_locations' is deprecated in favor of preprocess_joint_locs.py (which could be ported in here tbh)
        if not output_type in ['joint_locations', 'joint_thetas', 'raw_pkls']:
            raise Exception("output_type should be one of ['joint_locations', 'joint_thetas', 'raw_pkls']")

        # we need a body model to convert beta, theta and global translation into 3D joint locations.
        if output_type == 'joint_locations':
            # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            body_model = smplx.create(smplx_model_path,
                                      model_type='smplx',  ## smpl, smpl+h, or smplx?
                                      gender='neutral', ext='npz',  ## file format
                                      num_pca_comps=12,  ## MANO hand pose pca component
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
                                      batch_size=1  ## how many bodies in a batch?
                                      )
            body_model.eval()
            self.body_model = body_model

        self.output_type = output_type

    def __getitem__(self, idx):
        _in_frames_dicts, in_frames_fns, _pred_frames_dicts, pred_frames_fns = super().__getitem__(idx)

        in_data, pred_data = map(lambda fns: [load(fn) for fn in fns], [in_frames_fns, pred_frames_fns])
        # In event of failed file read, have arrays of appropriate shape but filled with nans - these training pairs
        # will be filtered out by our defined collate_fn in batching.
        if None in in_data or None in pred_data:
            # in_skels, pred_skels = nans_of_shape((self.in_frames, 21, 3)), nans_of_shape((self.pred_frames, 21, 3))
            in_skels, pred_skels = None, None
        elif self.output_type == 'joint_thetas':  # .reshape(-1, 21, 3)
            in_skels, pred_skels = map(
                lambda datas: torch.stack([torch.Tensor(data['body_pose']) for data in datas], dim=0).reshape(-1, 21,
                                                                                                              3),
                [in_data, pred_data])
        elif self.output_type == 'joint_locations':
            try:
                in_joint_locations, pred_joint_locations = normalized_joint_locations(in_data, pred_data, self.body_model)
            except Exception as e:
                print(f'exception: {e}, args {e.args}')
                print(f'idx was: {idx}')
                print(f'in_data, pred_data: {in_data}, {pred_data}')
                print(f'{in_frames_fns}, {pred_frames_fns}')
                raise e

        if self.output_type == 'raw_pkls':
            return (idx, (in_frames_fns, in_data), (pred_frames_fns, pred_data))
        if self.output_type == 'joint_thetas':
            return (idx, in_skels, pred_skels)
        elif self.output_type == 'joint_locations':
            return (idx, in_joint_locations, pred_joint_locations)

        return (idx, in_skels, pred_skels) if not self.verbose else (
        idx, (in_frames_fns, in_data), (pred_frames_fns, pred_data))


proxDatasetJoints = proxDatasetSkeleton  # backwards compatibility

class proxDatasetProximityMap(Dataset):
    def __init__(self, fittings_dir, depth_dir, calibration_dir):
        self.skelDataset = proxDatasetSkeleton(root_dir=fittings_dir, output_type='raw_pkls', in_frames=1, pred_frames=0)
        depth_dir = Path(depth_dir)
        self.depthDataset = DatasetBase(root_dir=depth_dir, search_prefix='Depth', extra_prefix='', in_frames=1, pred_frames=0)
        self.proj = Projection(calib_dir=calibration_dir)

    def __len__(self):
        return len(self.depthDataset)
    def __getitem__(self, idx):
        in_frames_dicts, in_frames_fns_img, _, _ = self.depthDataset.__getitem__(idx)
        depth_img = [cv2.imread(fn, flags=-1).astype(float) for fn in in_frames_fns_img][0]
        depth_img = cv2.flip(depth_img, 1)

        (idx, (in_frames_fns_skel, in_data), (pred_frames_fns, pred_data)) = self.skelDataset.__getitem__(idx)
        skeleton_dict = in_data[0]
        return in_frames_fns_img[0], in_frames_fns_skel[0], depth_img, skeleton_dict



class proxDatasetImages(Dataset):
    def __init__(self, root_dir='./', in_frames=10, pred_frames=5):
        self.root_dir = root_dir

        self.in_frames = in_frames
        self.pred_frames = pred_frames

        self.datasets = {}
        self.datasets['color'] = DatasetBase(root_dir=root_dir, in_frames=in_frames, pred_frames=pred_frames,
                                             search_prefix='Color', extra_prefix='')
        self.datasets['depth'] = DatasetBase(root_dir=root_dir, in_frames=in_frames, pred_frames=pred_frames,
                                             search_prefix='Depth', extra_prefix='')

        self.lengths = []
        for d in self.datasets.items():
            self.lengths.append(len(d))

        assert self.lengths == self.lengths[0] * len(self.lengths)

    def __len__(self):
        return self.lengths[0]

    def __getitem__(self, idx):
        in_frames_dicts, in_frames_fns, pred_frames_dicts, pred_frames_fns = self.datasets['Color'].__getitem__(idx)


#  doesn't exist. there's 02919, 02920, 02970 then 02950 weirdly
# try pd2.__getitem__(2098)  this is the offending sample
# FileNotFoundError: [Errno 2] No such file or directory: 'D:\\prox_data\\PROXD_attempt2\\PROXD\\MPH1Library_00145_01\\results\\s001_frame_02920__00.01.37.300\\000.pkl'

def my_collate(batch):
    # I think these are still np.arrays, will become tensors later
    batch = list(filter(
        # check that they exist and don't have nans??
        lambda triple: (triple[1] is not None) and (triple[2] is not None) and (
            not torch.any(torch.isnan(triple[1]))) and (not torch.any(torch.isnan(triple[2]))),
        batch
    ))
    return default_collate(batch)
