import copy
from glob import glob
import cv2
from matplotlib import scale
import numpy as np
import torch
from pytorch3d import transforms
import pickle
from pathlib import Path
import smplx
import tqdm
import open3d as o3d

def normalize_euler_angles(base_rot_matrix, euler_angles):
    old_rot_matrix = transforms.euler_angles_to_matrix(euler_angles, convention='XYZ')
    new_rot_matrix = old_rot_matrix @ base_rot_matrix.transpose(1, 2)
    new_euler_angles = transforms.matrix_to_euler_angles(new_rot_matrix, convention='XYZ')
    return new_euler_angles


def get_joint_locations(data_dict, body_model):
    betas, body_pose, global_orient, transl = extract_data(data_dict)
    out = body_model(return_joints=True, betas=betas, body_pose=body_pose, global_orient=global_orient, transl=transl)
    return out.joints[:, :21].squeeze().detach().cpu().numpy(), (out.vertices.detach().cpu().numpy().squeeze())


# TODO body_model is used here without definition - pass around with global or as param?
def normalized_joint_locations1(data_dict, base_rot_matrix, base_transl, body_model):
    betas, body_pose, euler_angles, transl = extract_data(data_dict)
    transl = transl - base_transl
    euler_angles = normalize_euler_angles(base_rot_matrix, euler_angles)

    out = body_model(return_joints=True, betas=betas, body_pose=body_pose, global_orient=euler_angles, transl=transl)
    joint_locs = out.joints[:, :25]  # 1, 25, 3
    return joint_locs


def normalized_joint_locations(in_data_dicts, pred_data_dicts, body_model):
    # TODO rotation normalisation only makes sense about the vector which aligns with gravity, or maybe not at all?
    _betas, _body_pose, base_euler_angles, base_transl = extract_data(in_data_dicts[-1])
    base_rot_matrix = transforms.euler_angles_to_matrix(base_euler_angles, convention='XYZ')

    in_joint_locations = torch.cat(
        [normalized_joint_locations1(data_dict, base_rot_matrix, base_transl, body_model) for data_dict in in_data_dicts], dim=0)
    pred_joint_locations = torch.cat(
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

def get_smplx_body_model(smplx_model_path):
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
    return body_model



# TODO what should this be?
BACKGROUND_PROX = 10.0

DEPTH_SCALE = 8 * 1000.

LIMBS = [
(0, 1),  # head_center -> neck
(1, 2),  # neck -> right_clavicle
(2, 3),  # right_clavicle -> right_shoulder
(3, 4),  # right_shoulder -> right_elbow
(4, 5),  # right_elbow -> right_wrist
(1, 6),  # neck -> left_clavicle
(6, 7),  # left_clavicle -> left_shoulder
(7, 8),  # left_shoulder -> left_elbow
(8, 9),  # left_elbow -> left_wrist
(1, 10),  # neck -> spine0
(10, 11),  # spine0 -> spine1
(11, 12),  # spine1 -> spine2
(12, 13),  # spine2 -> spine3
(13, 14),  # spine3 -> spine4
(14, 15),  # spine4 -> right_hip
(15, 16),  # right_hip -> right_knee
(16, 17),  # right_knee -> right_ankle
(14, 18),  # spine4 -> left_hip
(18, 19),  # left_hip -> left_knee
(19, 20),  # left_knee -> left_ankle
]

def scale_camera_dict(camera_dict, scale=DEPTH_SCALE):
    camera_dict['f'][0] *= scale
    camera_dict['f'][1] *= scale

    camera_dict['camera_mtx'][0][0] *= scale
    camera_dict['camera_mtx'][1][1] *= scale



def proximity_map(depth_map, skeleton_dict, body_model, depth_cam, alignment_cam):
    joint_locations, body_points =  get_joint_locations(skeleton_dict, body_model)
    
    skeleton = o3d.geometry.PointCloud()
    skeleton.points = o3d.utility.Vector3dVector(body_points)

    depth_cam = copy.deepcopy(depth_cam)
    scale_camera_dict(depth_cam)
    
    o3d_depth_map = o3d.geometry.Image(depth_map.astype(np.float32))

    h, w = depth_map.shape
   
    # TODO figure out if depth_scale, depth_trunc necessary
    pcd = o3d.geometry.PointCloud.create_from_depth_image(
    o3d_depth_map,
    o3d.camera.PinholeCameraIntrinsic(w, h, depth_cam['f'][0], depth_cam['f'][1], depth_cam['c'][0], depth_cam['c'][1]),
    # depth_scale=1000.,

    )  # cam coordinate

    points = np.asarray(pcd.points)
    points[:, 2] /= DEPTH_SCALE
    pcd.points = o3d.utility.Vector3dVector(points)

    points_color_aligned = align_point_clouds(points, depth_cam, alignment_cam)
    pcd_color_aligned = o3d.geometry.PointCloud()
    pcd_color_aligned.points = o3d.utility.Vector3dVector(points_color_aligned)


    body_bps = point_proximity_map(points_color_aligned, joint_locations)
    


    # TODO human mask
    depth_mask_ind = np.where(depth_map.flatten() == 0.)[0]
    # where we have points in the point cloud
    depth_nomask_ind = np.asarray(list(set(range(h*w))-set(depth_mask_ind)))  # indices of nonmasked pixels
    # TODO set sky mask where depth_aligned > MAX_DEPTH.
    # depth_mask_sky_ind = ((depth_aligned.flatten() == 0) * (human_mask==False)).reshape(h*w)  # indices of sky

    body_bps_full = np.zeros([h * w])
    body_bps_full[depth_nomask_ind] = body_bps
    body_bps_full[depth_mask_ind] = BACKGROUND_PROX

    body_bps_full = body_bps_full.reshape((h, w))  # body bps feature map, [h, w]
    body_bps_full = body_bps_full.astype(np.float32)

    return pcd, pcd_color_aligned, skeleton, body_bps_full


def align_point_clouds(points, depth_cam, alignment_cam):
    R1 = np.array(depth_cam['R'])
    # equivalent seems
    # R1 = np.array(transforms.euler_angles_to_matrix(torch.Tensor(R), convention='XYZ'))
    R1, _ = cv2.Rodrigues(R1)
    # same
    T1 = np.array(depth_cam['T'])

    R2 = np.array(alignment_cam['R'])
    # equivalent seems
    # R1 = np.array(transforms.euler_angles_to_matrix(torch.Tensor(R), convention='XYZ'))
    R2, _ = cv2.Rodrigues(R2)
    # same
    T2 = np.array(alignment_cam['T'])


    points_color_aligned = (R2 @ (R1.T @ (points - T1).T)).T + T2
    return points_color_aligned



def point_proximity_map(points, joint_locations):

    A = joint_locations[np.asarray(LIMBS)[:, 0]]  # [n_limb, 3]
    B = joint_locations[np.asarray(LIMBS)[:, 1]]  # [n_limb, 3]
    n_pt = points.shape[0]
    n_limb = A.shape[0]

    A = np.tile(A, (n_pt, 1)).reshape(n_pt*n_limb, 3)  # [n_pt, n_limb, 3], n_pt=n_bps=scene_verts.shape[0]
    B = np.tile(B, (n_pt, 1)).reshape(n_pt*n_limb, 3)
    P = np.tile(points, n_limb).reshape(n_pt*n_limb, 3)

    AB = B - A
    AP = P - A
    BP = P - B

    temp_1 = np.multiply(AB, AP).sum(axis=-1)  # [n_pt, n_limb]
    temp_2 = np.multiply(-AB, BP).sum(axis=-1)  # [n_pt, n_limb]
    mask_1 = np.where(temp_1 <= 0)[0]   # angle between AB and AP >= 90
    mask_2 = np.where((temp_1 > 0) * (temp_2 <= 0))[0]  # angle between AB and AP < 90 and angle between BA and BP >= 90
    mask_3 = np.where((temp_1 > 0) * (temp_2 > 0))[0]   # angle between AB and AP < 90 and angle between BA and BP < 90
    if len(mask_1) + len(mask_2) + len(mask_3) != n_pt*n_limb:
        print('[distance calculation] num of verts does not match!')

    dist_1 = np.sqrt(np.sum((P[mask_1]-A[mask_1])**2, axis=-1))  # [n_mask_1]
    dist_2 = np.sqrt(np.sum((P[mask_2]-B[mask_2])**2, axis=-1))  # [n_mask_2]

    x = np.multiply(AB[mask_3], AP[mask_3]).sum(axis=-1) / np.multiply(AB[mask_3], AB[mask_3]).sum(axis=-1)  # [n_mask_3]
    x = x.repeat(3).reshape(-1,3)
    C = x * AB[mask_3] + A[mask_3]  # C: [n_mask_3, 3], the projected point of P on line segment AB
    dist_3 = np.sqrt(np.sum((P[mask_3]-C)**2, axis=-1))  # n_mask_3

    dist = np.zeros(n_pt*n_limb)
    dist[mask_1] = dist_1
    dist[mask_2] = dist_2
    dist[mask_3] = dist_3
    dist = dist.reshape(n_pt, n_limb)  # [n_pt, n_limb], distance from each point in scene verts to each limb
    body_bps = np.min(dist, axis=-1)   # [n_pt]
    
    return body_bps
