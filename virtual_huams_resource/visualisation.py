import time
from typing import List
import numpy as np

from utils import cam2world_conv, world2cam_conv # for data manipulation
print('numpy: %s' % np.__version__) # print version
import math # to help with data reshaping of the data

import numpy as np
import torch
#import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# from sklearn.model_selection import train_test_split
import tqdm
import matplotlib.pyplot as plt
import logging

import open3d as o3d

# from IPython.core.interactiveshell import InteractiveShell
# InteractiveShell.ast_node_interactivity = "all"

# import os
# os.chdir('../')
# import os.path as osp

from pose_gru import PoseGRU_inputFC2
from benji_prox_dataloader import *


# os.environ['PYOPENGL_PLATFORM']= 'osmesa'  # our offscreen renderer will use OSMesa not Pyglet which is good as we don't have an active display manager on euler
# # https://github.com/mkocabas/VIBE/issues/47 seems like maybe egl won't work
# os.environ['MUJOCO_GL']= 'osmesa'

import pyrender
import PIL.Image as pil_img

import trimesh

# def render_skeleton_on_image(joint_locations: np.ndarray, img: np.ndarray, mesh_color=(1.0, 1.0, 0.9, 1.0)):

#     H, W = 1080, 1920

#     r = pyrender.OffscreenRenderer(viewport_width=W,
#                         viewport_height=H,
#                         point_size=10.0)

#     camera_center = np.array([951.30, 536.77])  # Idk why these, do be quite close to halfway point
#     camera_pose = np.eye(4)
#     camera_pose = np.array([1.0, -1.0, -1.0, 1.0]).reshape(-1, 1) * camera_pose
#     camera = pyrender.camera.IntrinsicsCamera(
#         fx=1060.53, fy=1060.38,
#         cx=camera_center[0], cy=camera_center[1])
#     light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)

#     material = pyrender.MetallicRoughnessMaterial(
#         metallicFactor=0.0,
#         alphaMode='OPAQUE',
#         baseColorFactor=mesh_color)
#     # body_mesh = pyrender.Mesh.from_trimesh(
#     #     out_mesh, material=material)

#     skel_mesh = pyrender.Mesh.from_points(joint_locations, colors=[mesh_color]*len(joint_locations))

#     H2, W2, _ = img.shape
#     assert(H2 == H and W2 == W)

#     scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0],
#                         ambient_light=(0.3, 0.3, 0.3))
#     scene.add(camera, pose=camera_pose)
#     scene.add(light, pose=camera_pose)
#     # for node in light_nodes:
#     #     scene.add_node(node)

#     scene.add(skel_mesh, 'mesh')

#     r = pyrender.OffscreenRenderer(viewport_width=W,
#                                 viewport_height=H,
#                                 point_size=1.0)
#     body_color, body_depth = r.render(scene, flags=pyrender.RenderFlags.RGBA)
#     body_color = body_color.astype(np.float32)  / 255.0

#     # valid_mask = (color[:, :, -1] > 0)[:, :, np.newaxis]
#     # input_img = img
#     # output_img = (color[:, :, :-1] * valid_mask +
#     #               (1 - valid_mask) * input_img)

#     valid_mask = (body_color > 0)
#     input_img = img
#     output_img = (body_color * valid_mask + (1-valid_mask)*np.flip(input_img, axis=1)/255.0)  # image is reversed for whatever reason

#     return body_color, body_depth, output_img


def points_to_mesh(vertices: np.ndarray, mesh_color=(1.0, 1.0, 0.9, 1.0)):
    out_mesh = pyrender.Mesh.from_points(vertices, colors=[mesh_color]*len(vertices))
    return out_mesh

def prox_camera_pyrender_renderer_and_scene():
    H, W = 1080, 1920  # TODO these better be the same as image shape

    r = pyrender.OffscreenRenderer(viewport_width=W,
                            viewport_height=H) 

    camera_center = np.array([951.30, 536.77])  # Idk why these, do be quite close to halfway point
    camera_pose = np.eye(4)
    camera_pose = np.array([1.0, -1.0, -1.0, 1.0]).reshape(-1, 1) * camera_pose
    camera = pyrender.camera.IntrinsicsCamera(
        fx=1060.53, fy=1060.38,
        cx=camera_center[0], cy=camera_center[1])
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)

    scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0],
                        ambient_light=(0.3, 0.3, 0.3))
    scene.add(camera, pose=camera_pose)
    scene.add(light, pose=camera_pose)

    return r, scene

def render_meshes_on_image(meshes: List[pyrender.Mesh], img: np.ndarray = None, point_size=1.0):
    r, scene = prox_camera_pyrender_renderer_and_scene()

    ## rendering body
    if img is not None:
        H2, W2, _ = img.shape
        assert(H2 == r.viewport_height and W2 == r.viewport_width) # sanity check? maybe not necessary

    for i, mesh in enumerate(meshes):
        scene.add(mesh, f'mesh{i}')

    r.point_size=point_size
    body_color, body_depth = r.render(scene)
    body_color = body_color.astype(np.float32)  / 255.0

    # valid_mask = (color[:, :, -1] > 0)[:, :, np.newaxis]
    # input_img = img
    # output_img = (color[:, :, :-1] * valid_mask +
    #               (1 - valid_mask) * input_img)

    valid_mask = (body_color > 0)
    input_img = img if img is not None else np.zeros(body_color.shape)
    output_img = (body_color * valid_mask + (1-valid_mask)*np.flip(input_img, axis=1)/255.0)  # image is reversed for whatever reason

    return body_color, body_depth, output_img    


def skels_and_background_to_video(in_skels: torch.Tensor, pred_skels: torch.Tensor, fut_skels: torch.Tensor, images: List[np.array]):
    """all skels should be in rgb world coordinate to match images. all these tensors are ? x 25 x 3"""
    gt_skels_color = (0.3, 1.0, 0.3, 1.0)
    pred_skels_color = (1.0, 0.3, 0.3, 1.0)
    point_size=15.0
    output_images = []
    for skel, img in zip(in_skels, images[:len(in_skels)]):
        mesh = points_to_mesh(skel.cpu().detach().numpy(), gt_skels_color)
        body_color, body_depth, out_img = render_meshes_on_image(meshes=[mesh], img=img, point_size=point_size)
        output_images.append(out_img)

    for (pred_skel, fut_skel), img in zip(zip(pred_skels, fut_skels), images[len(in_skels):]):
        pred_mesh = points_to_mesh(pred_skel.cpu().detach().numpy(), pred_skels_color)
        fut_mesh = points_to_mesh(fut_skel.cpu().detach().numpy(), gt_skels_color)
        body_color, body_depth, out_img = render_meshes_on_image(meshes=[pred_mesh, fut_mesh], img=img, point_size=point_size)
        output_images.append(out_img)

    return output_images


def predict_and_visualise(gru: PoseGRU_inputFC2, in_skels_world: torch.Tensor, fut_skels_world: torch.Tensor, images: List[np.array], cam2world, guided=True):
    """predict skeletons with model. Transform all skels from world to rgb space. plot camera rgb frames with skeleton joint point clouds
    rendered on top for each frame (green points for ground truth and red for predicted joints)
    
    all skeleton tensors of shape ? x 25 x 3 e.g. 15 x 25 x 3 and 30 x 25 x 3 (in vs fut)"""

    pelvis = in_skels_world[0, 0, :]
    in_skels_world_norm = in_skels_world - pelvis
    fut_skels_world_norm = fut_skels_world - pelvis

    if guided:
        cur_state, pred_skels_world = gru.forward_prediction_guided(in_skels_world_norm.unsqueeze(0),fut_skels_world_norm.unsqueeze(0))
    else:
        cur_state, pred_skels_world = gru.forward_prediction(in_skels_world_norm.unsqueeze(0), out_seq_len=len(fut_skels_world))
    pred_skels_world = pred_skels_world + pelvis
    pred_skels_world = pred_skels_world.squeeze()  # (30, 25, 3)
    
    in_skels, pred_skels, fut_skels = map(
        lambda skels_world: world2cam_conv(skels_world, cam2world),
        [in_skels_world, pred_skels_world, fut_skels_world]
    )

    output_images = skels_and_background_to_video(in_skels, pred_skels, fut_skels, images)
    return output_images


def predict_and_visualise_transformer(transformer, in_skels_world: torch.Tensor, fut_skels_world: torch.Tensor,
                          images: List[np.array], cam2world, guided=True):
    """predict skeletons with model. Transform all skels from world to rgb space. plot camera rgb frames with skeleton joint point clouds
    rendered on top for each frame (green points for ground truth and red for predicted joints)

    all skeleton tensors of shape ? x 25 x 3 e.g. 15 x 25 x 3 and 30 x 25 x 3 (in vs fut)

    """

    pelvis = in_skels_world[0, 0, :]
    in_skels_world_norm = in_skels_world - pelvis
    fut_skels_world_norm = fut_skels_world - pelvis
    in_skels_world_norm = torch.flatten(in_skels_world_norm.unsqueeze(0), start_dim=2)
    fut_skels_world_norm = torch.flatten(fut_skels_world_norm.unsqueeze(0), start_dim=2)
    tgt_mask = transformer.get_tgt_mask(fut_skels_world_norm.shape[1]).to(in_skels_world.device)

    if guided:
        pred_skels_world = transformer(in_skels_world_norm, fut_skels_world_norm, tgt_mask=tgt_mask).squeeze(0).reshape(-1, 25, 3)
    else:
        pred_skels_world = transformer.forward_predict(in_skels_world_norm, pred_frames=fut_skels_world.shape[0]).squeeze(0).reshape(
            -1, 25, 3)

    pred_skels_world = pred_skels_world + pelvis
    pred_skels_world = pred_skels_world.squeeze()  # (30, 25, 3)

    in_skels, pred_skels, fut_skels = map(
        lambda skels_world: world2cam_conv(skels_world, cam2world),
        [in_skels_world, pred_skels_world, fut_skels_world]
    )

    output_images = skels_and_background_to_video(in_skels, pred_skels, fut_skels, images)
    return output_images

def predict_and_visualise_transformer_unguided(transformer, in_skels_world: torch.Tensor, fut_skels_world: torch.Tensor,
                          images: List[np.array], cam2world, pred_frames=30):
    pelvis = in_skels_world[0, 0, :]
    in_skels_world_norm = in_skels_world - pelvis

    in_skels_world_norm = torch.flatten(in_skels_world_norm.unsqueeze(0), start_dim=2)

    pred_skels_world = transformer.forward_predict(in_skels_world_norm, pred_frames=pred_frames).squeeze(0).reshape(-1, 25, 3)
    pred_skels_world = pred_skels_world + pelvis
    pred_skels_world = pred_skels_world.squeeze()  # (30, 25, 3)

    in_skels, pred_skels, fut_skels = map(
        lambda skels_world: world2cam_conv(skels_world, cam2world),
        [in_skels_world, pred_skels_world, fut_skels_world]
    )
    output_images = skels_and_background_to_video(in_skels, pred_skels, fut_skels, images)
    return output_images

# TODO deprecate this function apart from when viewing smplx model hmmm
def render_mesh_on_image(vertices: np.ndarray, faces: np.ndarray = None, img: np.ndarray = None, mesh_color=(1.0, 1.0, 0.9, 1.0), point_size=1.0):
    """
    faces: is None then actually renders points for a skeleton - makes point_size=17.0
    img: background image optional or None otherwise
    """
  
    out_mesh = trimesh.Trimesh(vertices, faces, process=False) if faces is not None else \
        pyrender.Mesh.from_points(vertices, colors=[mesh_color]*len(vertices))


    # common
    H, W = 1080, 1920  # TODO these better be the same as image shape

    r, scene = prox_camera_pyrender_renderer_and_scene()

    material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.0,
        alphaMode='OPAQUE',
        baseColorFactor=mesh_color)
    if faces is not None:
        out_mesh = pyrender.Mesh.from_trimesh(
            out_mesh, material=material)

    ## rendering body
    if img is not None:
        H2, W2, _ = img.shape
        assert(H2 == H and W2 == W)


    scene.add(out_mesh, 'mesh')

    body_color, body_depth = r.render(scene)
    body_color = body_color.astype(np.float32)  / 255.0

    # valid_mask = (color[:, :, -1] > 0)[:, :, np.newaxis]
    # input_img = img
    # output_img = (color[:, :, :-1] * valid_mask +
    #               (1 - valid_mask) * input_img)

    valid_mask = (body_color > 0)
    input_img = img if img is not None else np.zeros(body_color.shape)
    output_img = (body_color * valid_mask + (1-valid_mask)*np.flip(input_img, axis=1)/255.0)  # image is reversed for whatever reason

    return body_color, body_depth, output_img

def smplx_and_background_to_video(images: List[np.ndarray], smplx_dicts: List[dict],\
     body_model, output_folder=None, mesh_color=(1.0, 1.0, 0.9, 1.0)):
    num_frames = len(images)
    assert num_frames == len(smplx_dicts)

    outputs = []
    for frame in range(num_frames):
        image, smplx_dict = images[frame], smplx_dicts[frame]

        betas = torch.Tensor(smplx_dict['betas'])
        body_pose = torch.Tensor(smplx_dict['body_pose'])
        global_orient= torch.Tensor(smplx_dict['global_orient'])
        transl=torch.Tensor(smplx_dict['transl'])
        out = body_model(return_joints=True, betas=betas, body_pose=body_pose, global_orient=global_orient, transl=transl)
        joints = out.joints[:, :25].squeeze()
        vertices = out.vertices.detach().cpu().numpy().squeeze()
    
        body_color, body_depth, output_img = render_mesh_on_image(vertices, body_model.faces, image, mesh_color=mesh_color)

        if output_folder:
            os.makedirs(output_folder, exist_ok=True)
            output_img2 = pil_img.fromarray((output_img * 255).astype(np.uint8))
            output_img2.save(os.path.join(output_folder, f'frame{frame}.png'))
        outputs.append(output_img)

    return outputs

# def skels_and_background_to_video(in_skels: torch.Tensor, pred_skels: torch.Tensor, fut_skels: torch.Tensor, images: List[np.array]):
#     """all skels should be in rgb world coordinate to match images. all these tensors are ? x 25 x 3"""
#     gt_skels_color = (0.3, 1.0, 0.3, 1.0)
#     pred_skels_color = (1.0, 0.3, 0.3, 1.0)
#     point_size=10.0
#     output_images = []
#     for skel, img in zip(in_skels, images[:len(in_skels)]):
#         body_color, body_depth, out_img = render_mesh_on_image(vertices = skel.detach().numpy(),\
#             img=img, mesh_color=gt_skels_color, point_size=point_size)
#         output_images.append(out_img)

#     for (pred_skel, fut_skel), img in zip(zip(pred_skels, fut_skels), images[len(in_skels):]):
#         body_color, body_depth, out_img_pred = render_mesh_on_image(vertices = pred_skel.detach().numpy(),\
#             img=img, mesh_color=pred_skels_color, point_size=point_size)

#         body_color, body_depth, out_img_fut = render_mesh_on_image(vertices = fut_skel.detach().numpy(),\
#             img=img, mesh_color=gt_skels_color, point_size=point_size)
#         output_images.append((out_img_pred + out_img_fut)/2)

#     return output_images

def joint_locs_and_background_to_video(images: List[np.ndarray], joint_locations: List[np.ndarray], translation: np.ndarray,\
     output_folder=None, mesh_color=(1.0, 1.0, 0.9, 1.0)):
    num_frames = len(images)
    assert num_frames == len(joint_locations)
    joint_locations = joint_locations + translation




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

color_input = np.zeros([len(LIMBS), 3])
color_input[:, 0] = 1.0

def animate_skeleton(skeleton_frames):

        
    trans = np.eye(4)
    trans[:3, :3] = np.array([[0, 0, -1], [-1, 0, 0], [0, -1, 0]])
    rx = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    ry = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
    rz = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    # trans[:3, :3] = trans[:3, :3]@rz@rz@rz@ry
    # trans[:3, -1] = np.array([0, 0, -3])
    trans[:3, :3] = trans[:3, :3]@rz@rz@rz
    trans[:3, -1] = np.array([5, 0, 3])

    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    vis.add_geometry(mesh_frame)
    outputs = []
    for t in range(skeleton_frames.shape[0]):  
        print(t)
        skeleton_input = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(skeleton_frames[t]),
            lines=o3d.utility.Vector2iVector(LIMBS))
        skeleton_input.colors = o3d.utility.Vector3dVector(color_input)

        vis.add_geometry(skeleton_input)

        ctr = vis.get_view_control()
        cam_param = ctr.convert_to_pinhole_camera_parameters()
        cam_param = update_cam(cam_param, trans)
        ctr.convert_from_pinhole_camera_parameters(cam_param)

        vis.poll_events()
        vis.update_renderer()
        outputs.append(vis.capture_screen_float_buffer())
        vis.remove_geometry(skeleton_input)



def update_cam(cam_param, trans):
    cam_R = np.transpose(trans[:-1, :-1])
    cam_T = -trans[:-1, -1:]
    cam_T = np.matmul(cam_R, cam_T)  # T is applied in the rotated coord
    cam_aux = np.array([[0, 0, 0, 1]])
    mat = np.concatenate([cam_R, cam_T], axis=-1)
    mat = np.concatenate([mat, cam_aux], axis=0)
    cam_param.extrinsic = mat
    return cam_param

# plt.imshow(image)
# plt.figure()
# plt.imshow(color)
# plt.figure()
# plt.imshow(depth)
# plt.figure()
# plt.imshow(torch.tensor(image))
# plt.figure()
# plt.imshow(torch.tensor(image).detach().cpu().numpy())
# plt.imshow(output_img)

# output_img2 = pil_img.fromarray((output_img * 255).astype(np.uint8))
# output_img2.save(out_img_fn)

# ##redering body+scene
# body_mesh = pyrender.Mesh.from_trimesh(
#     out_mesh, material=material)
# static_scene = trimesh.load(osp.join(scene_dir, scene_name + '.ply'))
# trans = np.linalg.inv(cam2world)
# static_scene.apply_transform(trans)

# static_scene_mesh = pyrender.Mesh.from_trimesh(
#     static_scene)

# scene = pyrender.Scene()
# scene.add(camera, pose=camera_pose)
# scene.add(light, pose=camera_pose)

# scene.add(static_scene_mesh, 'mesh')
# scene.add(body_mesh, 'mesh')

# r = pyrender.OffscreenRenderer(viewport_width=W,
#                                viewport_height=H)
# color, _ = r.render(scene)
# color = color.astype(np.float32) / 255.0
# img = pil_img.fromarray((color * 255).astype(np.uint8))
# # img.save(body_scene_rendering_fn)

# input: scene_dir (directory of the .ply file)
# input: skeleton (np.array 25 x 3)
def visualize_skeleton_in_point_cloud(scene_dir, skeleton):
    scene_point_cloud_input = o3d.io.read_point_cloud(scene_dir)
    skeleton_input = o3d.geometry.LineSet(
            o3d.utility.Vector3dVector(skeleton),
            o3d.utility.Vector2iVector(LIMBS))

    vis = o3d.visualization.Visualizer()
    vis.add_geometry(scene_point_cloud_input)
    vis.add_geometry(skeleton_input)
    
    return vis;
# create a Visualizer object and run the following to visualize:
# vis.run()
# vis.destroy_window()

# input: scene_dir (directory of the .ply file)
# input: skeleton_list (list of tensors of size 1 x 25 x 3)
def visualize_skeleton_sequences_in_point_cloud(scene_dir, skeleton_list):
    skeleton_iter = iter(skeleton_list)
    # update function for callback
    def update(vis):
        skeleton = next(skeleton_iter)          # render first frame
        # skeleton = skeleton.detach().numpy()
        # skeleton = skeleton.astype(np.float64)
        # skeleton = skeleton.squeeze()
        # print(skeleton)
        skeleton_input = o3d.geometry.LineSet(
                o3d.utility.Vector3dVector(skeleton),
                o3d.utility.Vector2iVector(LIMBS))
        ctrl = vis.get_view_control()
        cam_param = ctrl.convert_to_pinhole_camera_parameters()
        vis.clear_geometries()
        vis.add_geometry(scene_point_cloud)
        vis.add_geometry(skeleton_input)
        ctrl.convert_from_pinhole_camera_parameters(cam_param)
        vis.poll_events()
        vis.update_renderer()
        vis.run()

    scene_point_cloud = o3d.io.read_point_cloud(scene_dir)
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.register_key_callback(65, update)       # press A
    vis.add_geometry(scene_point_cloud)

    skeleton = skeleton_list[0]
    # skeleton = skeleton_list[0].detach().numpy()          # render first frame
    # skeleton = skeleton.astype(np.float64)
    # skeleton = skeleton.squeeze()
    skeleton_input = o3d.geometry.LineSet(
                o3d.utility.Vector3dVector(skeleton),
                o3d.utility.Vector2iVector(LIMBS))
    vis.add_geometry(skeleton_input)
    vis.run()
    
    return ;
# run the function and press 'A' to proceed to the next frame