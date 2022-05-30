import os

import numpy as np  # for data manipulation

np.random.seed(0)
import math  # to help with data reshaping of the data

import numpy as np
import torch
import json

torch.manual_seed(0)

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from simple_transformer import PoseTransformer

import tqdm
import matplotlib.pyplot as plt
import logging
from visualisation import predict_and_visualise_transformer

# from IPython.core.interactiveshell import InteractiveShell
# InteractiveShell.ast_node_interactivity = "all"


import sys

sys.path.append('../')
sys.path.append('../../src')

from pose_gru import PoseGRU_inputFC2
from benji_prox_dataloader import *

print(f'cuda availability: {torch.cuda.is_available()}')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'device: {device}')

name = "Transformer_joints_15_30_3fps_30_05_1613"

import wandb
_ = wandb.init(settings=wandb.Settings(), project="transformer_viz", entity="vh-motion-pred", name=name)


# root_dir = "../data_new/"
smplx_model_path = 'C:\\Users\\xiyi\\projects\\semester_project\\smplify-x\\smplx_model\\models\\'
viz_folder = '../viz_prox_validation/'

batch_size = 15
in_frames = 15
pred_frames = 30
frame_jump = 10
window_overlap_factor = 450
lr = 0.0001
n_iter = 200
save_every = 40
num_workers = 0
max_loss = 10000  # This is dangerous but stops ridiculous updates?
best_val_loss = np.inf
bsub_command = 'bsub -n 20 -R "rusage[mem=16384,ngpus_excl_p=1]" python rnn_gru_joints_worldnorm.py'

save_folder = 'saves'
os.makedirs(save_folder, exist_ok=True)

save_path = os.path.join(save_folder, name + '_epoch{epoch}_bn{batchnum}.pt')
save_path.format(epoch=3, batchnum=5)

val_areas = ['BasementSittingBooth', 'N3OpenArea']

pd_train = proxDatasetSkeleton(root_dir='../data_train/PROXD/', in_frames=in_frames, pred_frames=pred_frames, \
                         output_type='raw_pkls', smplx_model_path=smplx_model_path, frame_jump=10,
                         window_overlap_factor=150, extra_prefix='joints_worldnorm.pkl')

pd_valid = proxDatasetSkeleton(root_dir='../data_valid/PROXD/', in_frames=in_frames, pred_frames=pred_frames, \
                             output_type='raw_pkls', smplx_model_path=smplx_model_path, frame_jump=10,
                             window_overlap_factor=8, extra_prefix='joints_worldnorm.pkl')


pd_train.sequences = [seq for seq in pd_train.sequences if not any([area in seq[0] for area in val_areas])]
pd_valid.sequences = [seq for seq in pd_valid.sequences if any([area in seq[0] for area in val_areas])]

pdc = DatasetBase(root_dir='../data_train/recordings/', in_frames=in_frames, pred_frames=pred_frames,
                  search_prefix='Color', extra_prefix='', frame_jump=frame_jump,
                  window_overlap_factor=window_overlap_factor)

pdc.align(pd_valid)
pd_valid.align(pdc)



def my_collate2(batch):
    # I think these are still np.arrays, will become tensors later

    batch = list(filter(
        # check that they exist and don't have nans??
        lambda triple: (triple[1] is not None) and (triple[2] is not None),
        batch
    ))
    try:
        batch = [(triple[0], torch.stack(triple[1][1]).squeeze(), torch.stack(triple[2][1]).squeeze()) for triple in
                 batch]

        batch = list(filter(
            lambda triple: (not torch.any(torch.isnan(triple[1]))) and (not torch.any(torch.isnan(triple[2])))
                           and (not torch.any(torch.gt(triple[1], 100)))
                           and (not torch.any(torch.gt(triple[2], 100)))
                           and (not torch.any(torch.lt(triple[1], -100)))
                           and (not torch.any(torch.lt(triple[2], -100))), batch
        ))
        return default_collate(batch)

    except Exception as e:
        print(f'batch: {[(triple[0], triple[1].shape, triple[2].shape) for triple in batch]}')
        print(e, e.args)
        raise e


dataloader_train = DataLoader(pd_train, batch_size=batch_size,
                              shuffle=True, num_workers=0, collate_fn=my_collate2)

dataloader_valid = DataLoader(pd_valid, batch_size=batch_size,
                              shuffle=True, num_workers=0, collate_fn=my_collate2)
criterion = nn.MSELoss()

model = PoseTransformer(num_tokens=25*3).to(device)
# model = TimeSeriesTransformer().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

writer = SummaryWriter()

# wandb.config = {
#     "learning_rate": lr,
#     "epochs": n_iter,
#     "batch_size": batch_size,
#     "in_frames": in_frames,
#     "pred_frames": pred_frames,
#     "frame_jump": frame_jump,
#     "window_overlap_factor": window_overlap_factor,
#     "max_loss": max_loss,
#     "num_workers": num_workers
# }

# with open('config.json', 'w') as file:
#     file.write(json.dumps(dict(wandb.config)))
#
# wandb.save('config.json')

idx_counter = 0
last_fn = None
for epoch in range(n_iter):
    losses = []
    losses_rep = []

    losses_valid = []
    losses_rep_valid = []
    for i, (idx, in_skels, fut_skels) in (pbar := tqdm.tqdm(enumerate(dataloader_train), total=len(dataloader_train))):
        in_skels = in_skels.to(device)
        fut_skels = fut_skels.to(device)

        pelvis = in_skels[:, 0, 0, :].unsqueeze(1).unsqueeze(1)
        in_skels = in_skels - pelvis
        in_skels_cpy = in_skels.clone()
        in_skels = torch.flatten(in_skels, start_dim=2)

        fut_skels = fut_skels - pelvis
        fut_skels = torch.flatten(fut_skels, start_dim=2)
        tgt = torch.cat((in_skels[:, -1, :].unsqueeze(1), fut_skels[:, :-1, :]), dim=1)

        tgt_mask = model.get_tgt_mask(fut_skels.shape[1]).to(device)

        optimizer.zero_grad()

        pred_frames = fut_skels.shape[1]
        batch_len = fut_skels.shape[0]
        # print(f'batch_len: {batch_len}')  # maybe something's wrong but I do get about avg 13 batchlen not 15 :( crummy files?

        pred_skels = model(in_skels, tgt, tgt_mask=tgt_mask)

        loss = criterion(pred_skels, fut_skels)
        loss.backward()
        if loss.item() < max_loss:
            optimizer.step()

        rep_pred = in_skels_cpy[:, -1, :, :]
        a = rep_pred.detach().cpu().numpy()

        a = np.tile(a, (fut_skels.shape[1], 1, 1, 1))
        rep_pred = torch.Tensor(a).transpose(0, 1).to(device)
        rep_pred_shape = rep_pred.shape
        rep_pred = rep_pred.reshape(rep_pred_shape[0], rep_pred_shape[1], rep_pred_shape[2] * rep_pred_shape[3])

        loss_rep = criterion(rep_pred, fut_skels)
        losses_rep.append(loss_rep.item())

        losses.append(loss.item())

        # wandb.log({'MSEloss': loss, 'rep_pred_MSEloss': loss_rep})

        pbar.set_description(
            f"avg last 20 loss: {np.mean(list(filter(lambda x: x < max_loss, losses[-20:]))):.4f} ")

        writer.add_scalar('Loss', losses[-1], idx_counter)
        writer.add_scalar('Loss_rep', losses_rep[-1], idx_counter)
        if i % save_every == (save_every - 1):
            fn = save_path.format(epoch=epoch, batchnum=i)
            torch.save({
                'epoch': epoch,
                'batch_num': i,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, fn)
            # wandb.save(fn)
            if last_fn:
                os.remove(last_fn)
            last_fn = fn

        idx_counter += 1

        wandb.log({'batch_train_loss': loss.item(),
                   'batch_train_reploss': loss_rep.item()})
    print(f'end epoch {epoch}: total mean loss: {np.mean(list(filter(lambda x: x < max_loss, losses)))}, '
          f'mean rep loss: {np.mean(list(filter(lambda x: x < max_loss, losses_rep)))}')

    wandb.log({'epoch_train_loss': np.mean(list(filter(lambda x: x < max_loss, losses))),
               'epoch_train_reploss': np.mean(list(filter(lambda x: x < max_loss, losses_rep)))})

    # print('visualization')
    # # for i in range(3):
    # for idx in tqdm.tqdm(range(len(pd_valid))):
    #     # idx = np.random.randint(pd_valid.bounds[-1])
    #     try:
    #         (_, (_, in_skels), (_, fut_skels)) = pd_valid.__getitem__(idx)
    #         _, in_frames_fns, _, pred_frames_fns = pdc.__getitem__(idx)
    #         print(in_frames_fns, pred_frames_fns)
    #         in_imgs = [np.array(cv2.imread(fn)) for fn in in_frames_fns]
    #         fut_imgs = [np.array(cv2.imread(fn)) for fn in pred_frames_fns]
    #     except Exception as e:  # some skel fn None so get idx None, None. Or image files unreadable etc.
    #         continue
    #     if in_frames_fns == [] or pred_frames_fns == []:
    #         continue
    #
    #     if in_skels is not None and fut_skels is not None:
    #         in_skels_world = torch.cat(in_skels)
    #         fut_skels_world = torch.cat(fut_skels)
    #     if torch.any(torch.isnan(in_skels_world)) or torch.any(torch.isnan(fut_skels_world)):
    #         continue
    #
    #     # start, seq_idx = get_start_idx(idx, pd_valid.bounds, pd_valid.start_jump)
    #     print(in_frames_fns)
    #     scene_name = in_frames_fns[0].split('\\')[3].split('_')[0]
    #     video_name = in_frames_fns[0].split('\\')[3]
    #     if os.path.exists(os.path.join(viz_folder, video_name, 'epoch' + str(epoch))):
    #         continue
    #     print(idx, video_name)
    #     with open(f'{root_dir}/cam2world/{scene_name}.json') as file:
    #         cam2world = np.array(json.load(file))
    #         cam2world = torch.from_numpy(cam2world).float()
    #
    #     images = in_imgs + fut_imgs
    #     img_fns = in_frames_fns + pred_frames_fns
    #
    #     output_images = predict_and_visualise_transformer(model, in_skels_world.to(device), fut_skels_world.to(device), images, cam2world.to(device))
    #     images_down = [
    #         cv2.resize((img * 255).astype(np.uint8), dsize=(int(img.shape[1] / 5), int(img.shape[0] / 5))) for img
    #         in output_images]
    #     if not os.path.exists(os.path.join(viz_folder, video_name)):
    #         os.makedirs(os.path.join(viz_folder, video_name))
    #     if not os.path.exists(os.path.join(viz_folder, video_name, 'epoch' + str(epoch))):
    #         os.makedirs(os.path.join(viz_folder, video_name, 'epoch' + str(epoch)))
    #     for i, img in enumerate(images_down):
    #         cv2.imwrite(os.path.join(viz_folder, video_name, 'epoch' + str(epoch), img_fns[i].split('\\')[-1]), img)
    #
    #     # wandb.log({f'val_image_seq{i}': [wandb.Image(img) for img in images_down]})

    print('validation loss')
    with torch.no_grad():
        for i, (idx, in_skels, fut_skels) in (
        pbar := tqdm.tqdm(enumerate(dataloader_valid), total=len(dataloader_valid))):
            in_skels = in_skels.to(device)
            fut_skels = fut_skels.to(device)

            pelvis = in_skels[:, 0, 0, :].unsqueeze(1).unsqueeze(1)
            in_skels = in_skels - pelvis
            in_skels_cpy = in_skels.clone()
            in_skels = torch.flatten(in_skels, start_dim=2)

            fut_skels = fut_skels - pelvis
            fut_skels = torch.flatten(fut_skels, start_dim=2)
            tgt = torch.cat((in_skels[:, -1, :].unsqueeze(1), fut_skels[:, :-1, :]), dim=1)

            tgt_mask = model.get_tgt_mask(fut_skels.shape[1]).to(device)
            pred_frames = fut_skels.shape[1]
            batch_len = fut_skels.shape[0]
            # print(f'batch_len: {batch_len}')  # maybe something's wrong but I do get about avg 13 batchlen not 15 :( crummy files?

            pred_skels = model(in_skels, tgt, tgt_mask=tgt_mask)

            loss = criterion(pred_skels, fut_skels)

            rep_pred = in_skels_cpy[:, -1, :, :]
            a = rep_pred.detach().cpu().numpy()

            a = np.tile(a, (fut_skels.shape[1], 1, 1, 1))
            rep_pred = torch.Tensor(a).transpose(0, 1).to(device)
            rep_pred_shape = rep_pred.shape
            rep_pred = rep_pred.reshape(rep_pred_shape[0], rep_pred_shape[1], rep_pred_shape[2] * rep_pred_shape[3])

            loss_rep = criterion(rep_pred, fut_skels)
            losses_rep_valid.append(loss_rep.item())

            losses_valid.append(loss.item())
            pbar.set_description(
                f"avg last 20 loss: {np.mean(list(filter(lambda x: x < max_loss, losses_valid[-20:]))):.4f} ")
            wandb.log({'batch_val_loss': loss.item(),
                       'batch_val_reploss': loss_rep.item()})
        print(
            f'end epoch {epoch}: total mean validation loss: {np.mean(list(filter(lambda x: x < max_loss, losses_valid)))}, '
            f'mean rep loss: {np.mean(list(filter(lambda x: x < max_loss, losses_rep_valid)))}')
        if np.mean(list(filter(lambda x: x < max_loss, losses_valid))) < best_val_loss:
            best_val_loss = np.mean(list(filter(lambda x: x < max_loss, losses_valid)))
            torch.save({
                'epoch': epoch,
                'batch_num': i,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, os.path.join(save_folder, 'transformer_best_model.pt'))

        wandb.log({'epoch_val_loss': np.mean(list(filter(lambda x: x < max_loss, losses_valid))),
                   'epoch_val_reploss': np.mean(list(filter(lambda x: x < max_loss, losses_rep_valid)))})

plt.plot(losses)
print(losses[-4:])
