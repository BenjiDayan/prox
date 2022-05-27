# This script will preprocess out joint locations from our smplx data pickles, as well as
# converting the 25, 3 points into the world reference frame via cam2world


import json
import numpy as np


print('numpy: %s' % np.__version__) # print version
np.random.seed(0)
import math # to help with data reshaping of the data

import numpy as np
import torch
torch.manual_seed(0)

import tqdm
import matplotlib.pyplot as plt
import logging

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

import sys
sys.path.append('../')
sys.path.append('../../src')

from pose_gru import PoseGRU_inputFC2
from benji_prox_dataloader import *
from utils import normalized_joint_locations_world # for data manipulation




root_dir = "/cluster/scratch/bdayan/prox_data"
smplx_model_path='/cluster/home/bdayan/prox/prox/models_smplx_v1_1/models/'

batch_size = 15
in_frames=15
pred_frames=30
frame_jump=5
window_overlap_factor=5
lr=0.0001
n_iter = 10
save_every=40
max_loss = 5. # This is dangerous but stops ridiculous updates? 


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pd = proxDatasetJoints(root_dir=root_dir + '/PROXD', in_frames=in_frames, pred_frames=pred_frames, \
                       output_type='joint_locations', smplx_model_path=smplx_model_path, frame_jump=frame_jump, window_overlap_factor=window_overlap_factor)



def load(fn):
    try:
        with open(fn, 'rb') as file:
            return pickle.load(file)
    except FileNotFoundError:
        return None


def get_joints(fn, body_model, cam2world):
    smplx_dict = load(fn)
    joints = normalized_joint_locations_world(smplx_dict, body_model, cam2world)
    return joints



for seq in pd.sequences:
    print(seq[0])
    try:
        area_name = seq[0][:seq[0].index('_')]
        with open(root_dir + '/cam2world/' + area_name + '.json') as file:
            cam2world = np.array(json.load(file))
            cam2world = torch.from_numpy(cam2world).float().to(device)

    except Exception as e:
        print(e, e.args)
        print(seq[0])
        continue

    for fn_dict in tqdm.tqdm(seq[1], total=len(seq)):
        try:
            fn = fn_dict['fn']
        except Exception as e:
            print(e, e.args)
            print(fn_dict)
            continue

        fn = Path(fn)
        new_fn = str(fn.parent / 'joints_worldnorm.pkl')

        try:
            joints = get_joints(fn, pd.body_model, cam2world)
        except Exception as e:
            print(e, e.args)
            joints = None
        
        with open(new_fn, 'wb') as file:
            pickle.dump(joints, file)
        
