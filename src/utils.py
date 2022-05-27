from glob import glob
import torch
from pytorch3d import transforms
import pickle
from pathlib import Path
import smplx
import tqdm

import numpy as np
from tqdm import tqdm
import scipy.ndimage as img_fn
from gta_utils import LIMBS
import logging
import datetime
import os
import json


def normalize_euler_angles(base_rot_matrix, euler_angles):
    old_rot_matrix = transforms.euler_angles_to_matrix(euler_angles, convention='XYZ')
    new_rot_matrix = old_rot_matrix @ base_rot_matrix.transpose(1, 2)
    new_euler_angles = transforms.matrix_to_euler_angles(new_rot_matrix, convention='XYZ')
    return new_euler_angles


# TODO body_model is used here without definition - pass around with global or as param?
def normalized_joint_locations1(data_dict, base_rot_matrix, base_transl, body_model):
    betas, body_pose, euler_angles, transl = extract_data(data_dict)
    transl = transl - base_transl
    euler_angles = normalize_euler_angles(base_rot_matrix, euler_angles)

    out = body_model(return_joints=True, betas=betas, body_pose=body_pose, global_orient=euler_angles, transl=transl)
    joint_locs = out.joints[:, :25]  # 1, 25, 3
    return joint_locs


def normalized_joint_locations(in_data_dicts, pred_data_dicts, body_model):
    _betas, _body_pose, base_euler_angles, base_transl = extract_data(in_data_dicts[-1])
    base_rot_matrix = transforms.euler_angles_to_matrix(base_euler_angles, convention='XYZ')

    in_joint_locations = torch.concat(
        [normalized_joint_locations1(data_dict, base_rot_matrix, base_transl, body_model) for data_dict in in_data_dicts], dim=0)
    pred_joint_locations = torch.concat(
        [normalized_joint_locations1(data_dict, base_rot_matrix, base_transl, body_model) for data_dict in pred_data_dicts], dim=0)

    return in_joint_locations, pred_joint_locations


def extract_data(data_dict):
    betas = torch.Tensor(data_dict['betas'])
    body_pose = torch.Tensor(data_dict['body_pose'])
    global_orient = torch.Tensor(data_dict['global_orient'])
    transl = torch.Tensor(data_dict['transl'])
    return betas, body_pose, global_orient, transl

def l1_inpainting(f, mask, theta=0.001, maxIter=5000):
    u = np.array((1 - mask) * f)
    mask = np.pad(mask, pad_width=1, mode='edge')
    f = np.pad(f, pad_width=1, mode='edge')

    for i in range(maxIter):
        u_b = np.pad(u, pad_width=1, mode='edge')
        ux = img_fn.sobel(u, axis=1)
        uy = img_fn.sobel(u, axis=0)
        phi_prime = np.pad( 1 /np.sqrt(1 + (ux**2 +uy**2 ) /theta**2), pad_width=1, mode='edge')

        u_bp = np.roll(u_b, 1, axis=1)
        u_bn = np.roll(u_b, -1, axis=1)
        u_bu = np.roll(u_b, 1, axis=0)
        u_bd = np.roll(u_b, -1, axis=0)

        a_p = 0.5 *(phi_prime + np.roll(phi_prime, 1, axis=1))
        a_n = 0.5 *(phi_prime + np.roll(phi_prime, -1, axis=1))
        a_u = 0.5 *(phi_prime + np.roll(phi_prime, 1, axis=0))
        a_d = 0.5 *(phi_prime + np.roll(phi_prime, -1, axis=0))
        a_c = (a_p + a_n + a_u + a_d)

        u_c = ((1 - mask) * f + 1.0 * mask * (a_p * u_bp + a_n * u_bn + a_u * u_bu + a_d * u_bd)
               ) / (1 - mask + 1.0 * mask * a_c)
        u = np.copy(u_c[1:-1, 1:-1])
        
    return u