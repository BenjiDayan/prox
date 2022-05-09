from glob import glob
import torch
from pytorch3d import transforms
import pickle
from pathlib import Path
import smplx
import tqdm


def normalize_euler_angles(base_rot_matrix, euler_angles):
    old_rot_matrix = transforms.euler_angles_to_matrix(euler_angles, convention='XYZ')
    new_rot_matrix = old_rot_matrix @ base_rot_matrix.transpose(1, 2)
    new_euler_angles = transforms.matrix_to_euler_angles(new_rot_matrix, convention='XYZ')
    return new_euler_angles

# TODO body_model is used here without definition - pass around with global or as param?
def normalized_joint_locations1(data_dict, base_rot_matrix, base_transl):
    betas, body_pose, euler_angles, transl = extract_data(data_dict)
    transl = transl - base_transl
    euler_angles = normalize_euler_angles(base_rot_matrix, euler_angles)
    
    out = body_model(return_joints=True, betas=betas, body_pose=body_pose, global_orient=euler_angles, transl=transl)
    joint_locs = out.joints[:, :25]  # 1, 25, 3
    return joint_locs


def normalized_joint_locations(in_data_dicts, pred_data_dicts):
    _betas, _body_pose, base_euler_angles, base_transl = extract_data(in_data_dicts[-1])
    base_rot_matrix = transforms.euler_angles_to_matrix(base_euler_angles, convention='XYZ')

    in_joint_locations = torch.concat([normalized_joint_locations1(data_dict, base_rot_matrix, base_transl) for data_dict in in_data_dicts], dim=0)
    pred_joint_locations = torch.concat([normalized_joint_locations1(data_dict, base_rot_matrix, base_transl) for data_dict in pred_data_dicts], dim=0)

    return in_joint_locations, pred_joint_locations


def extract_data(data_dict):
    betas = torch.Tensor(data_dict['betas'])
    body_pose = torch.Tensor(data_dict['body_pose'])
    global_orient= torch.Tensor(data_dict['global_orient'])
    transl=torch.Tensor(data_dict['transl'])
    return betas, body_pose, global_orient, transl