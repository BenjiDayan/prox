from glob import glob
import torch
from benji_prox_dataloader import proxDataset
from pytorch3d import transforms
import pickle
from pathlib import Path
import smplx
import tqdm

from utils import *


if __name__ == '__main__':

    # root_dir = 'D:/prox_data/PROXD_attempt2/PROXD'
    # smplx_model_path='../models_smplx_v1_1/models/'

    root_dir = '/cluster/scratch/bdayan/prox_data/PROXD_attempt2/PROXD'
    smplx_model_path = '/cluster/home/bdayan/models_smplx_v1_1/models/'

    in_frames = 10
    pred_frames = 5
    batch_size = 15

    print(root_dir)
    pd = proxDataset(root_dir, in_frames=in_frames, pred_frames=pred_frames, output_type='raw_pkls', smplx_model_path=smplx_model_path)


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


    output_dir = pd.root_dir / '../../joint_locations'  # D:/prox_data/PROXD_attempt2/PROXD/../../

    print(output_dir)

    errors = []
    for i in tqdm.tqdm(range(len(pd)), total=len(pd)):
        idx, in_data, pred_data = pd.__getitem__(i)
        in_data_dicts, pred_data_dicts =  in_data[1], pred_data[1]

        last_in_fn_path = Path(in_data[0][-1])  # this corresponds to last frame of the inputs, i.e. T-1 for predciting T, T+1, ...
        new_fn = output_dir / last_in_fn_path.relative_to(pd.root_dir)  
        try:
            in_joint_locations, pred_joint_locations = normalized_joint_locations(in_data_dicts, pred_data_dicts)
        except Exception as e:  # I think either e.g. nans messing things up or perhaps the missing pkl error :((
            errors.append((in_data[0], pred_data[0]))
        save_dict = {'in_fns': in_data[0], 'pred_fns': pred_data[0], 'in_joint_locations': in_joint_locations, 'pred_joint_locations': pred_joint_locations}

        new_fn.parent.mkdir(parents=True, exist_ok=True)
        # print(f'new_fn: {new_fn}')
        with open(str(new_fn), 'wb') as file:
            pickle.dump(save_dict, file)

    print(f'errors: \n{errors}')
