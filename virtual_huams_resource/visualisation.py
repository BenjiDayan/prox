from typing import List
import numpy as np # for data manipulation
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

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

import os
os.chdir('../')
import os.path as osp

from pose_gru import PoseGRU_inputFC2
from benji_prox_dataloader import *


os.environ['PYOPENGL_PLATFORM']= 'osmesa'  # our offscreen renderer will use OSMesa not Pyglet which is good as we don't have an active display manager on euler
# https://github.com/mkocabas/VIBE/issues/47 seems like maybe egl won't work
os.environ['MUJOCO_GL']= 'osmesa'

import pyrender
import PIL.Image as pil_img

import trimesh

def render_mesh_on_image(vertices: np.ndarray, faces: np.ndarray, img: np.ndarray):
    """
    for data in in_data[1]:
        betas = torch.Tensor(data['betas'])
        body_pose = torch.Tensor(data['body_pose'])
        global_orient= torch.Tensor(data['global_orient'])
        transl=torch.Tensor(data['transl'])
        out = smplx_neutral(return_joints=True, betas=betas, body_pose=body_pose, global_orient=global_orient, transl=transl)
        joints = out.joints[:, :25].squeeze()
        joint_locations.append(joints)
        vertices.append(out.vertices.detach().cpu().numpy().squeeze())

    vertices = vertices[0]
    faces = smplx_neutral.faces

    render_mesh_on_image
    """
    out_mesh = trimesh.Trimesh(vertices, faces, process=False)

    W=1920
    H=1080
    r = pyrender.OffscreenRenderer(viewport_width=W,
                                viewport_height=H,
                                point_size=1.0)


    # common
    H, W = 1080, 1920
    camera_center = np.array([951.30, 536.77])  # Idk why these, do be quite close to halfway point
    camera_pose = np.eye(4)
    camera_pose = np.array([1.0, -1.0, -1.0, 1.0]).reshape(-1, 1) * camera_pose
    camera = pyrender.camera.IntrinsicsCamera(
        fx=1060.53, fy=1060.38,
        cx=camera_center[0], cy=camera_center[1])
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)

    material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.0,
        alphaMode='OPAQUE',
        baseColorFactor=(1.0, 1.0, 0.9, 1.0))
    body_mesh = pyrender.Mesh.from_trimesh(
        out_mesh, material=material)

    ## rendering body

    H, W, _ = img.shape

    scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0],
                        ambient_light=(0.3, 0.3, 0.3))
    scene.add(camera, pose=camera_pose)
    scene.add(light, pose=camera_pose)
    # for node in light_nodes:
    #     scene.add_node(node)

    scene.add(body_mesh, 'mesh')

    r = pyrender.OffscreenRenderer(viewport_width=W,
                                viewport_height=H,
                                point_size=1.0)
    body_color, body_depth = r.render(scene, flags=pyrender.RenderFlags.RGBA)
    body_color = body_color.astype(np.float32)  / 255.0

    # valid_mask = (color[:, :, -1] > 0)[:, :, np.newaxis]
    # input_img = img
    # output_img = (color[:, :, :-1] * valid_mask +
    #               (1 - valid_mask) * input_img)

    valid_mask = (body_color > 0)
    input_img = img
    output_img = (body_color * valid_mask + (1-valid_mask)*np.flip(input_img, axis=1)/255.0)  # image is reversed for whatever reason

    return body_color, body_depth, output_img

def smplx_and_background_to_video(images: List[np.ndarray], smplx_dicts: List[dict], body_model, output_folder=None):
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
    
        body_color, body_depth, output_img = render_mesh_on_image(vertices, body_model.faces, image)

        if output_folder:
            os.makedirs(output_folder, exist_ok=True)
            output_img2 = pil_img.fromarray((output_img * 255).astype(np.uint8))
            output_img2.save(os.path.join(output_folder, f'frame{frame}.png'))
        outputs.append(output_img)

    return outputs


    plt.imshow(image)
    plt.figure()
    plt.imshow(color)
    plt.figure()
    plt.imshow(depth)
    plt.figure()
    plt.imshow(torch.tensor(image))
    plt.figure()
    plt.imshow(torch.tensor(image).detach().cpu().numpy())
    plt.imshow(output_img)

    output_img2 = pil_img.fromarray((output_img * 255).astype(np.uint8))
    output_img2.save(out_img_fn)

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