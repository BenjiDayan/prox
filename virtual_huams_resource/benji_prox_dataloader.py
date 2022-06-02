import math
import os
from typing import List
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

from utils import normalized_joint_locations, proximity_map, get_smplx_body_model


def load(fn):
    if not fn:  # e.g. fn set to None in dataset alignment
        return None
    try:
        with open(fn, 'rb') as file:
            return pickle.load(file)
    except FileNotFoundError:
        return None


def nans_of_shape(shape):
    out = np.empty(shape)
    out[:] = np.nan
    return out


def get_start_idx(idx, bounds, num_frames):
    seq_idx = np.digitize(idx, bounds)
    assert seq_idx < len(bounds), "idx too big"
    idx_in_seq = idx - (bounds[seq_idx - 1] if seq_idx > 0 else 0)
    start = idx_in_seq * num_frames
    return start, seq_idx

# TODO why do these exist?
# '/PROXD/MPH1Library_00145_01/results/s001_frame_01945__00.01.04.801/000.pkl' example pkl file with massive transl, global_orient
class DatasetBase(Dataset):
    def __init__(self, root_dir='./', in_frames=10, pred_frames=5, search_prefix='results', extra_prefix='000.pkl', frame_jump=1, window_overlap_factor=2):
        """
        in_frames: number of frames from which to predict future frames
        pred_frames: number of future frames to predict
        frame_jump: difference in frame number. If 1 then sample every frame. If 5 then sample every 5th frame
  search for our files, which have names like
            /cluster/scratch/bdayan/prox_data/PROXD/N3OpenArea_00158_01/results/s001_frame_00006__00.00.00.148/000.pkl
            others are ...s001_frame_00006__00.00.00.148.png hence extra_prefix might be ''
        """
        self.root_dir = Path(root_dir)

        scenes = glob.glob(str(self.root_dir / '*'))
        scenes = [Path(scene) for scene in scenes]
        scenes_dir = {scene.name: list(map(Path, glob.glob(str(scene / search_prefix / '*')))) for scene in scenes}
        # for stem, paths in scenes_dir.items():
        #     verified_paths = []
        #     for path in paths:
        #         path = str(path / (extra_prefix if extra_prefix else ''))
        #         if os.path.exists(path):


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
        self.frame_jump = frame_jump
        self.window_length = self.tot_frames * self.frame_jump
        self.window_overlap_factor=window_overlap_factor
        self.start_jump = math.ceil(self.window_length/self.window_overlap_factor)  # subsequent start indices for each sequence

        length = len(self)  # initialise self.bounds

    def __len__(self):
        seq_lens = [len(fns_dict) for stem, fns_dict in self.sequences]
        # make sure all the windows fit inside.
        self.bounds = np.array([(seq_len - (self.window_length - self.start_jump)) // self.start_jump for seq_len in seq_lens])  # e.g.
        if not (np.all(self.bounds >= 1)):
            print("sequence has insufficient frames for one training input")  # sanity check
        self.bounds = np.cumsum(self.bounds)
        return self.bounds[-1]

    def __getitem__(self, idx):
        start, seq_idx = get_start_idx(idx, self.bounds, self.start_jump)

        in_frames_dicts = self.sequences[seq_idx][1][start:start + self.in_frames*self.frame_jump:self.frame_jump]
        pred_frames_dicts = self.sequences[seq_idx][1][start + self.in_frames*self.frame_jump:start + self.tot_frames*self.frame_jump:self.frame_jump]
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
            self.body_model = get_smplx_body_model(smplx_model_path)

        self.output_type = output_type

    def __getitem__(self, idx):
        _in_frames_dicts, in_frames_fns, _pred_frames_dicts, pred_frames_fns = super().__getitem__(idx)


        in_data, pred_data = map(lambda fns: [load(fn) for fn in fns], [in_frames_fns, pred_frames_fns])
        # In event of failed file read, have arrays of appropriate shape but filled with nans - these training pairs
        # will be filtered out by our defined collate_fn in batching.
        if None in in_data or None in pred_data:
            # in_skels, pred_skels = nans_of_shape((self.in_frames, 21, 3)), nans_of_shape((self.pred_frames, 21, 3))
            return (idx, None, None)
        elif self.output_type == 'joint_thetas':  # .reshape(-1, 21, 3)
            try:
                in_skels, pred_skels = map(
                    lambda datas: torch.stack([torch.Tensor(data['body_pose']) for data in datas], dim=0).reshape(-1, 21,
                                                                                                                3),
                    [in_data, pred_data])
            except Exception as e:
                print(f'exception: {e}, args: {e.args}')
                return (idx, None, None)
        elif self.output_type == 'joint_locations':
            try:
                in_joint_locations, pred_joint_locations = normalized_joint_locations(in_data, pred_data, self.body_model)
            except Exception as e:
                print(f'exception: {e}, args {e.args}')
                print(f'idx was: {idx}')
                print(f'in_data, pred_data: {in_data}, {pred_data}')
                print(f'{in_frames_fns}, {pred_frames_fns}')
                # raise e
                return (idx, None, None)

        if self.output_type == 'raw_pkls':
            return (idx, (in_frames_fns, in_data), (pred_frames_fns, pred_data))
        if self.output_type == 'joint_thetas':
            return (idx, in_skels, pred_skels)
        elif self.output_type == 'joint_locations':
            return (idx, in_joint_locations, pred_joint_locations)



proxDatasetJoints = proxDatasetSkeleton  # backwards compatibility

class proxDatasetProximityMap(DatasetBase):
    def __init__(self, fittings_dir, depth_dir, calibration_dir, smplx_model_path=None, **kwargs):
        self.skelDataset = proxDatasetSkeleton(root_dir=fittings_dir, output_type='raw_pkls', **kwargs)
        depth_dir = Path(depth_dir)
        self.depthDataset = DatasetBase(root_dir=depth_dir, search_prefix='Depth', extra_prefix='', **kwargs)
        self.proj = Projection(calib_dir=calibration_dir)

        self.skelDataset.align(self.depthDataset)
        self.depthDataset.align(self.skelDataset)

        self.body_model = get_smplx_body_model(smplx_model_path)
        

    def __len__(self):
        return len(self.depthDataset)


    def depth_and_skel_data_to_proximity_map(self, depth_fns: List[str], skel_datas: List[dict]):
        depth_imgs = frame_fns_to_images(depth_fns)
        proximity_map_data = []
        for depth_map, skel_dict in zip(depth_imgs, skel_datas):
            try:
                proximity_map_data.append(proximity_map(depth_map, skel_dict, self.body_model, self.proj.depth_cam, self.proj.color_cam)[-1])
            except Exception as e:
                print(e, e.args)
                proximity_map_data.append(None)
        
        return proximity_map_data

    def __getitem__(self, idx):
        in_frames_dicts, in_frames_fns, pred_frames_dicts, pred_frames_fns = self.depthDataset.__getitem__(idx)
        (idx, (in_skels_fns, in_data), (pred_skels_fns, pred_data)) = self.skelDataset.__getitem__(idx)

        try:
            in_prox_maps, pred_prox_maps = self.depth_and_skel_data_to_proximity_map(in_frames_fns, in_data), self.depth_and_skel_data_to_proximity_map(pred_frames_fns, pred_data)
        except Exception as e:
            return None, None
        return in_prox_maps, pred_prox_maps
        


def frame_fns_to_images(fns: List[str]):
    imgs = [cv2.imread(fn, flags=-1).astype(float) for fn in fns]
    return [cv2.flip(img, 1) for img in imgs]


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
    try:
        return default_collate(batch)
    except Exception as e:
        print(f'batch: {[(triple[0], triple[1].shape, triple[2].shape) for triple in batch]}')
        print(e, e.args)
        raise e
