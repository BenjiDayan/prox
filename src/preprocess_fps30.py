import argparse
import os
import pickle
import sys
import cv2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from gta_utils import LIMBS, read_depthmap
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm       # some fancy visual effect
import glob
from utils import *
import copy
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, default='/GTA-1M/FPS-30')
parser.add_argument('--sequence_id', type=str, default='2020-05-20-21-13-13_resize')
parser.add_argument('--save_root', type=str, default='/GTA-1M/FPS-30/preprocessed_data')

# output 2D proximity maps and many other things in /GTA-1M/FPS-30/preprocessed_data

args = parser.parse_args()


if __name__ == '__main__':
    ##################### set up parameters #########################
    MAX_DEPTH = 20.0
    h, w = 256, 448
    # h, w = 1080, 1920
    scale = 1080/h
    fps = 30
    depth_inpaint_itr = 500

    ##################### join paths #########################
    save_feature_img_path = os.path.join(args.save_root, args.sequence_id, 'bps_feature_img')
    save_feature_npy_path = os.path.join(args.save_root, args.sequence_id, 'bps_feature_npy')
    save_depth_img_path = os.path.join(args.save_root, args.sequence_id, 'depth_inpaint_img')
    save_depth_npy_path = os.path.join(args.save_root, args.sequence_id, 'depth_inpaint_npy')
    save_rgb_path = os.path.join(args.save_root, args.sequence_id, 'rgb_img_input')

    info = pickle.load(open(os.path.join(args.data_root, args.sequence_id, 'info_frames.pickle'), 'rb'))
    info_npz = np.load(os.path.join(args.data_root, args.sequence_id, 'info_frames.npz'))

    ##################### create directories #########################
    if not os.path.exists(save_feature_img_path):
        os.makedirs(save_feature_img_path)
    if not os.path.exists(save_feature_npy_path):
        os.makedirs(save_feature_npy_path)
    if not os.path.exists(save_depth_img_path):
        os.makedirs(save_depth_img_path)
    if not os.path.exists(save_depth_npy_path):
        os.makedirs(save_depth_npy_path)
    if not os.path.exists(save_rgb_path):
        os.makedirs(save_rgb_path)

    # base images
    rgb_list = glob.glob(os.path.join(args.data_root, args.sequence_id, '*.jpg'))       # collect paths of all .jpg images under the directory
    rgb_list.sort()
    n_frame = int(rgb_list[-1][-9:-4]) + 1  # total frame number


    vis = o3d.visualization.Visualizer()
    vis.create_window(width=w, height=h, visible=True)

    ##################### construct point clouds for a sequence of frames #########################
    seq_cnt = 0
    start_frame = fps - 1       # skip the first fsp-1 frames
    end_frame = n_frame - fps*4     # skip the last fps*4 frames
    # todo: step = fps*1 if enlarge training size
    for frame_N in tqdm(range(start_frame, end_frame, fps*3)):    # N_th frame: to get inpainted depth map --> reference frame
        global_pcd = o3d.geometry.PointCloud()

        start = max(0, frame_N-fps*6) if fps==5 else max(5, frame_N-fps*6)      # include the past fps*6 frames
        end = frame_N + fps*4     # include the future fps*4 frames
        step = 3 * fps // 5     # step (@Di: don't know why this step is used)

        # @Benjamin: We go for a range of a number of frames centered around frame_N - fps*6 to frame_N + fps*4 :O
        # I don't understand this. E.g. frame_N = 500, fps=30 gives frame_cnt: 320, 338, 356, ..., 608
        # Oh I think it's about getting a local point cloud for the current short time period.
        # @Di: Agree.
        for frame_cnt in range(start, end, step):   # 10s
            img_path = os.path.join(args.data_root, args.sequence_id, '{:05d}'.format(frame_cnt) + '.jpg')
            depth_path = os.path.join(args.data_root, args.sequence_id, '{:05d}'.format(frame_cnt) + '.png')
            human_mask_path = os.path.join(args.data_root, args.sequence_id, '{:05d}'.format(frame_cnt) + '_id.png')

            ##################### read and clean data #########################
            img = cv2.imread(img_path)      # read RGB image [h,w,3]
            img = img[:, :, ::-1]  # BGR --> RGB (@Di: don't know how this works)
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)

            # @Di: why do we need this parameter?
            infot = info[frame_cnt]
            cam_near_clip = infot['cam_near_clip']  # near/far: distances from the camera to start/stop rendering.
            if 'cam_far_clip' in infot.keys():
                cam_far_clip = infot['cam_far_clip']
            else:
                cam_far_clip = 800.
            depth = read_depthmap(depth_path, cam_near_clip, cam_far_clip)  # read depth image [1080, 1920, 1], in meters
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)   # [h,w]
            # plt.imshow(depth, cmap='plasma')
            # plt.show()

            # obtain the human mask
            joints_2d = info_npz['joints_2d']
            p = joints_2d[frame_cnt, 0]   # (x,y) coordinate of head joint
            id_map = cv2.imread(human_mask_path, cv2.IMREAD_ANYDEPTH)   # read object & human mask [1080, 1920]
            human_id = id_map[np.clip(int(p[1]), 0, 1079), np.clip(int(p[0]), 0, 1919)]   # scalar, uint8, human id
            # @Benjamin: If head is not in image?
            # @Di: This is just to extract the RGB value of human being in the image. np.clip() ensures that this will work
            # even if the human head is not in the image.
            mask_human = id_map == human_id     # extract human mask from object & human mask [1080, 1920], true/false, 1:person, 0:background
            mask_human = cv2.resize(mask_human.astype(np.uint8) , (w, h))
            kernel = np.ones((3, 3), np.uint8)
            mask_dilation = cv2.dilate(mask_human.astype(np.uint8), kernel, iterations=1)       # expand the mask a bit

            # @Di: why?
            depth[depth > MAX_DEPTH] = 0    # exclude points with large depth value
            depth[mask_dilation==True] = 0    # exclude person in the depth map


            #################### reconstruct point cloud from consecutive frames #######################
            cam_int = info_npz['intrinsics'][frame_cnt]  # camera intrinsic params [3,3]
            # retrieve focal lengths
            focal_length_x = cam_int[0, 0]
            focal_length_y = cam_int[1, 1]

            depth = np.expand_dims(depth, axis=-1)      # @Di: why is axis -1? [h,w,1]

            # from rgb image and human excluded depth map
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(img),
                o3d.geometry.Image(depth.astype(np.float32)),
                depth_scale=1.0,
                depth_trunc=MAX_DEPTH,
                convert_rgb_to_intensity=False,
            )
            # create coloured point cloud from rgbd image
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd_image,
                o3d.camera.PinholeCameraIntrinsic(w, h, focal_length_x/scale, focal_length_y/scale, (w/2), (h/2)),
            )  # cam coordinate
            
            depth_pts = np.asarray(pcd.points)     # [1920*1080-n_human_mask, 3], coordinate of each pixel in the depth map
            depth_pts_aug = np.hstack([depth_pts, np.ones([depth_pts.shape[0], 1])])        # euclidean coordinates to homogeneous coordinates
            cam_extr_ref = np.linalg.inv(info_npz['world2cam_trans'][frame_cnt])        # camera extrinsic params
            depth_pts = depth_pts_aug.dot(cam_extr_ref)[:, :3]      # camera coordinates to world coordinates
            pcd.points = o3d.utility.Vector3dVector(depth_pts)

            # we put the point cloud into the global point cloud
            global_pcd.points.extend(pcd.points)
            global_pcd.colors.extend(pcd.colors)


        # the global point cloud has all information (including occluded points behind human for the current frame)
        # therefore we extract a new current frame depth image from global point cloud to use instead of input data
        # depth image.
        ##################### capture depth/RGB image of cur_frame_N from point cloud ###############
        for cur_frame_N in range(frame_N-5, frame_N+1):
            vis.add_geometry(global_pcd)
            cam_ext = np.transpose(info_npz['world2cam_trans'][cur_frame_N])
            cam_int = info_npz['intrinsics'][cur_frame_N]
            ctr = vis.get_view_control()
            cam_param = ctr.convert_to_pinhole_camera_parameters()
            cam_param.intrinsic= o3d.camera.PinholeCameraIntrinsic(w, h, cam_int[0,0]/scale, cam_int[1,1]/scale, w/2-0.5, h/2-0.5)
            cam_param.extrinsic = cam_ext
            ctr.convert_from_pinhole_camera_parameters(cam_param)
            vis.poll_events()
            vis.update_renderer()


            # better than input depth and rgb images for current frame
            out_depth = np.asarray(vis.capture_depth_float_buffer())
            # plt.imshow(out_depth)
            # plt.show()
            out_image = np.asarray(vis.capture_screen_float_buffer())
            # plt.imshow(out_image)
            # plt.show()

            vis.remove_geometry(global_pcd)


            ######################## inpaint/save depth map of cur_frame_N ##############################
            infot = info[cur_frame_N]
            cam_near_clip = infot['cam_near_clip']
            if 'cam_far_clip' in infot.keys():
                cam_far_clip = infot['cam_far_clip']
            else:
                cam_far_clip = 800.
            depth = read_depthmap(os.path.join(args.data_root, args.sequence_id, '{:05d}'.format(cur_frame_N) + '.png'), cam_near_clip, cam_far_clip)
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)  # [h,w]
            mask_sky_cur_frame_N = depth > MAX_DEPTH        # sky mask

            joints_2d = info_npz['joints_2d']
            p = joints_2d[cur_frame_N, 0]  # (x,y) coordinate of head joint
            # human & object mask
            id_map = cv2.imread(os.path.join(args.data_root, args.sequence_id, '{:05d}'.format(cur_frame_N) + '_id.png'), cv2.IMREAD_ANYDEPTH)  # [1080, 1920]
            human_id = id_map[np.clip(int(p[1]), 0, 1079), np.clip(int(p[0]), 0, 1919)]
            mask_human_cur_frame_N = id_map == human_id     # human mask (true/false)
            mask_human_cur_frame_N = cv2.resize(mask_human_cur_frame_N.astype(np.uint8), (w, h))

            # (out_depth == 0) * (mask_human_cur_frame_N == false): sky
            # (out_depth == 0): human or sky
            depth_mask_human = (out_depth == 0) ^ (out_depth == 0) * (mask_human_cur_frame_N == False) # human mask
            depth_mask_sky = (out_depth == 0) * (mask_human_cur_frame_N == False) # sky
            depth_mask_human_ratio = np.sum(depth_mask_human)/(h*w) # ratio of human to the image
            # print(depth_mask_ratio)
            if depth_mask_human_ratio <= 0.01:      # if ratio of human is small enough
                if depth_mask_human_ratio == 0:     # if no human at all
                    inpainted_depth_map = out_depth     # this is the inpainted depth map
                    inpainted_depth_mask_human_ratio = 0
                else:       # if the ratio is relatively large
                    print('interpolating depth for frame {}'.format(cur_frame_N))       # interpolate the depth map
                    # @Di: where is this function l1_inpainting() from?
                    inpainted_depth_map = l1_inpainting(out_depth,
                                                        depth_mask_human,
                                                        maxIter=depth_inpaint_itr)
                    # plt.imshow(inpainted_depth_map)
                    # plt.show()
                    # (out_depth > 0): not human or sky
                    inpainted_depth_mask_human = inpainted_depth_map[depth_mask_human] <= np.min(out_depth[out_depth>0])
                    inpainted_depth_mask_human_ratio = np.sum(inpainted_depth_mask_human) / (h*w)

                if inpainted_depth_mask_human_ratio <= 0.001:       # if ratio of human is small enough
                    inpainted_depth_map = inpainted_depth_map.astype(np.float32)        # take as the interpolated inpainted depth map

                    ################### get scene point cloud of cur_frame_N ###################
                    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                        o3d.geometry.Image((np.asarray(out_image)*255).astype(np.uint8)),
                        o3d.geometry.Image(np.asarray(inpainted_depth_map.astype(np.float32))),
                        depth_scale=1.0,
                        depth_trunc=MAX_DEPTH,
                        convert_rgb_to_intensity=False,
                    )
                    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                        rgbd_image,
                        o3d.camera.PinholeCameraIntrinsic(w, h, cam_int[0,0]/scale, cam_int[1,1]/scale, w/2-0.5, h/2-0.5),
                    )  # cam coordinate

                    scene_verts = np.asarray(pcd.points)     # [h*w, 3], coordinate of each pixel in the depth map
                    scene_verts_aug = np.hstack([scene_verts, np.ones([scene_verts.shape[0], 1])])
                    cam_extr_ref = np.linalg.inv(info_npz['world2cam_trans'][cur_frame_N])
                    scene_verts = scene_verts_aug.dot(cam_extr_ref)[:, :3]      # camera to world transformation
                    pcd.points = o3d.utility.Vector3dVector(scene_verts) # world coordinate


                    if scene_verts.shape[0] == np.sum(inpainted_depth_map > 0):     # if all points in the depth map has a corresponding point in the point cloud
                        ######################## save rgb/depth image ##############################
                        print('save data for cur_frame_N = {}'.format(cur_frame_N))
                        seq_cnt += 1        # for every cur_frame_N, if 
                        plt.imsave('{}/seq_{:04d}_fr_{:05d}.png'.format(save_depth_img_path, seq_cnt, cur_frame_N), inpainted_depth_map)
                        np.save('{}/seq_{:04d}_fr_{:05d}.npy'.format(save_depth_npy_path, seq_cnt, cur_frame_N), inpainted_depth_map)

                        # save rgb img with person
                        img_cur_frame_N = cv2.imread(os.path.join(args.data_root, args.sequence_id, '{:05d}'.format(cur_frame_N) + '.jpg'))
                        img_cur_frame_N = img_cur_frame_N[:, :, ::-1]  # BGR --> RGB
                        img_cur_frame_N = cv2.resize(img_cur_frame_N, (w, h), interpolation=cv2.INTER_NEAREST)

                        rgb_image = copy.deepcopy(out_image)  # in [0,1]
                        rgb_image[mask_human_cur_frame_N == True] = img_cur_frame_N[mask_human_cur_frame_N == True] / 255.0 # scaled rgb value for human
                        rgb_image[mask_sky_cur_frame_N == True] = img_cur_frame_N[mask_sky_cur_frame_N == True] / 255.0 # scaled rgb value for sky
                        plt.imshow(rgb_image)
                        plt.imsave('{}/seq_{:04d}_fr_{:05d}.jpg'.format(save_rgb_path, seq_cnt, cur_frame_N), rgb_image)


                        ################### calculate body bps feature for all frames #######################
                        # some params (@Di: no idea why parameters are like this, maybe empirically determined)
                        start = cur_frame_N - fps//5 * 4
                        end = cur_frame_N + fps * 2 + 1
                        step = fps // 5
                        for cur_frame in range(start, end, step):
                            depth_flat = inpainted_depth_map.reshape(h*w)  # inpainted depth map
                            depth_mask_ind = np.where(depth_flat == 0)[0]  # indices of masked pixels (human or sky)
                            depth_nomask_ind = np.asarray(list(set(range(h*w))-set(depth_mask_ind)))  # indices of nonmasked pixels
                            depth_mask_sky_ind = ((out_depth == 0) * (mask_human_cur_frame_N==False)).reshape(h*w)  # indices of sky

                            # ############ distance to 21 joints ############
                            # nbrs = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm="ball_tree").fit(body_joints_3d)
                            # neigh_dist, neigh_ind = nbrs.kneighbors(scene_verts)
                            # body_bps = neigh_dist[:, 0]  # distance  [n_bps]  n_bps = h*w-n_human_mask
                            # print('max/min body bps feature value:', np.max(body_bps), np.min(body_bps))
                            # print(np.sum(body_bps))

                            ############ distance to skeleton (21 joints)  ############
                            # min distance from each pixel point P to a set of line segments AB
                            body_joints_3d = info_npz['joints_3d_world'][cur_frame]  # [21, 3] world coordinate
                            # 3D coordinates of the two end points of each limbs
                            A = body_joints_3d[np.asarray(LIMBS)[:, 0]]  # [n_limb, 3]
                            B = body_joints_3d[np.asarray(LIMBS)[:, 1]]  # [n_limb, 3]
                            n_pt = scene_verts.shape[0]
                            n_limb = A.shape[0]

                            A = np.tile(A, (n_pt, 1)).reshape(n_pt*n_limb, 3)  # [n_pt, n_limb, 3], n_pt=n_bps=scene_verts.shape[0]
                            B = np.tile(B, (n_pt, 1)).reshape(n_pt*n_limb, 3)
                            P = np.tile(scene_verts, n_limb).reshape(n_pt*n_limb, 3)

                            # compute distance from P to each limb
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
                            # print('max/min body bps feature value:', np.max(body_bps), np.min(body_bps))


                            ########### back to image plane, visualize bps feature map ###########
                            # print(cur_frame, cur_frame_N)
                            body_bps_full = np.zeros([h*w])
                            body_bps_full[depth_nomask_ind] = body_bps
                            body_bps_full[depth_mask_ind] = 0         # set masked pixels to 0
                            body_bps_full[depth_mask_sky_ind] = 100.0  # set sky pixels to 100
                            body_bps_full = body_bps_full.reshape((h, w))  # body bps feature map, [h, w]
                            body_bps_full = body_bps_full.astype(np.float32)
                            np.save('{}/seq_{:04d}_fr_{:05d}.npy'.format(save_feature_npy_path, seq_cnt, cur_frame), body_bps_full)

                            # plt.imshow(body_bps_full, cmap='plasma')
                            # plt.show()

                            body_bps_full[body_bps_full>10] = 10
                            fig = plt.imshow(body_bps_full, cmap='plasma', vmin=0, vmax=5)
                            plt.axis('off')
                            fig.axes.get_xaxis().set_visible(False)
                            fig.axes.get_yaxis().set_visible(False)
                            plt.imsave('{}/seq_{:04d}_fr_{:05d}.png'.format(save_feature_img_path, seq_cnt, cur_frame), body_bps_full, cmap='plasma')
                            # plt.show()


    print('total seq cnt:', seq_cnt)














