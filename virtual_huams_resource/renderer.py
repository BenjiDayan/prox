import pickle
import sys

import cv2
import numpy as np
import open3d as o3d
import smplx
import trimesh
from PIL import Image
from benji_dataloader import *

import time


def update_cam(cam_param, trans):
    cam_R = np.transpose(trans[:-1, :-1])
    cam_T = -trans[:-1, -1:]
    cam_T = np.matmul(cam_R, cam_T)  # T is applied in the rotated coord
    cam_aux = np.array([[0, 0, 0, 1]])
    mat = np.concatenate([cam_R, cam_T], axis=-1)
    mat = np.concatenate([mat, cam_aux], axis=0)
    cam_param.extrinsic = mat
    return cam_param

LIMBS = [(23, 15),
         (24, 15),
         (15, 22),
         (22, 12),
         # left arm
         (12, 13),
         (13, 16),
         (16, 18),
         (18, 20),
         # right arm
         (12, 14),
         (14, 17),
         (17, 19),
         (19, 21),
         # spline
         (12, 9),
         (9, 6),
         (6, 3),
         (3, 0),
         # left leg
         (0, 1),
         (1, 4),
         (4, 7),
         (7, 10),
         # right leg
         (0, 2),
         (2, 5),
         (5, 8),
         (8, 11)]

# frame0 = pred_fns[0]
# H, W = 1080, 1920
# camera_center = np.array([951.30, 536.77])
# camera_pose = np.eye(4)
# camera_pose = np.array([1.0, -1.0, -1.0, 1.0]).reshape(-1, 1) * camera_pose
# camera = pyrender.camera.IntrinsicsCamera(
#     fx=1060.53, fy=1060.38,
#     cx=camera_center[0], cy=camera_center[1])
# light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
# 
# body_model = smplx.create('/Users/xiyichen/Documents/semester_project/smplify-x/smplx_model/models/', model_type='smplx',
#                          gender='MALE', ext='npz',
#                          num_pca_comps=12,
#                          create_global_orient=True,
#                          create_body_pose=True,
#                          create_betas=True,
#                          create_left_hand_pose=True,
#                          create_right_hand_pose=True,
#                          create_expression=True,
#                          create_jaw_pose=True,
#                          create_leye_pose=True,
#                          create_reye_pose=True,
#                          create_transl=True
#                          )
# global_orient = frame0[0,:].reshape(1, 3)
# body_pose = frame0[1:22,:].reshape(1, 21, 3)
# jaw = frame0[22,:].reshape(1, 3)
# leye = frame0[23,:].reshape(1, 3)
# reye = frame0[24,:].reshape(1, 3)
# output = body_model(return_verts=True, global_orient=global_orient, body_pose=body_pose, jaw_pose=jaw, leye_pose=leye, reye_pose=reye)
# 
# vertices = output.vertices.detach().cpu().numpy().squeeze()
# print(vertices)
# body = trimesh.Trimesh(vertices, body_model.faces, process=False)
# 
# material = pyrender.MetallicRoughnessMaterial(
#             metallicFactor=0.0,
#             alphaMode='OPAQUE',
#             baseColorFactor=(1.0, 1.0, 0.9, 1.0))
# body_mesh = pyrender.Mesh.from_trimesh(
#     body, material=material)
# scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0],
#                                    ambient_light=(0.3, 0.3, 0.3))
# scene.add(camera, pose=camera_pose)
# scene.add(light, pose=camera_pose)
# 
# scene.add(body_mesh, 'mesh')
# img = cv2.imread('/Users/xiyichen/Desktop/1.jpg')[:, :, ::-1] / 255.0
# H, W, _ = img.shape
# print(W, H)
# r = pyrender.OffscreenRenderer(viewport_width=W,
#                                viewport_height=H,
#                                point_size=1.0)
# color, _ = r.render(scene, flags=pyrender.RenderFlags.RGBA)
# color = color.astype(np.float32)
# print(color.sum())
# 
# valid_mask = (color[:, :, -1] > 0)[:, :, np.newaxis]
# input_img = img
# output_img = (color[:, :, :-1] * valid_mask +
#               (1 - valid_mask) * input_img)
# 
# img = Image.fromarray((output_img * 255).astype(np.uint8))
# img.save('./test_output.png')


root_dir = "../joint_locations/"
smplx_model_path='/Users/xiyichen/Documents/semester_project/smplify-x/smplx_model/models/'
in_frames = 10
pred_frames = 5
batch_size = 5
pd = proxDatasetJoints(root_dir)

mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(mesh_frame)

trans = np.eye(4)
trans[:3, :3] = np.array([[0, 0, -1], [-1, 0, 0], [0, -1, 0]])
rx = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
ry = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
rz = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
# trans[:3, :3] = trans[:3, :3]@rz@rz@rz@ry
# trans[:3, -1] = np.array([0, 0, -3])
trans[:3, :3] = trans[:3, :3]@rz@rz@rz
trans[:3, -1] = np.array([5, 0, 3])

dataloader = DataLoader(pd, batch_size=batch_size,
                        shuffle=True, num_workers=0, collate_fn=my_collate)
future_frames = pd.__getitem__(0)[2]
past_frames = pd.__getitem__(100)[1]

body_joints_input = past_frames.detach().cpu().numpy()  # [T, 25/55, 3]

color_input = np.zeros([len(LIMBS), 3])
color_input[:, 0] = 1.0
color_rec = np.zeros([len(LIMBS), 3])
color_rec[:, 2] = 1.0
T = past_frames.shape[0]

for t in range(0, T):
    print(t)
    skeleton_input = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(body_joints_input[t]),
        lines=o3d.utility.Vector2iVector(LIMBS))
    skeleton_input.colors = o3d.utility.Vector3dVector(color_input)

    vis.add_geometry(skeleton_input)

    ctr = vis.get_view_control()
    cam_param = ctr.convert_to_pinhole_camera_parameters()
    cam_param = update_cam(cam_param, trans)
    ctr.convert_from_pinhole_camera_parameters(cam_param)

    vis.poll_events()
    vis.update_renderer()
    time.sleep(0.5)
    vis.remove_geometry(skeleton_input)