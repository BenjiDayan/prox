import os

import numpy as np  # for data manipulation
from torch.utils.data import DataLoader

np.random.seed(0)
import math  # to help with data reshaping of the data

import numpy as np
import torch
import json

torch.manual_seed(0)

import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from simple_transformer import PoseTransformer

import tqdm

import sys

sys.path.append('../')
sys.path.append('../../src')

from benji_prox_dataloader import *

print(f'cuda availability: {torch.cuda.is_available()}')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'device: {device}')

name = "Transformer_joints_5_10_5fps_15_06_1318"

# import wandb
# _ = wandb.init(settings=wandb.Settings(), project="transformer_new", entity="vh-motion-pred", name=name)


# root_dir = "../data_new/"
smplx_model_path = 'C:\\Users\\xiyi\\projects\\semester_project\\smplify-x\\smplx_model\\models\\'
viz_folder = '../viz_prox_validation/'

batch_size = 15
in_frames = 5
pred_frames = 10
frame_jump = 6
window_overlap_factor = 30
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
                         output_type='raw_pkls', smplx_model_path=smplx_model_path, frame_jump=6,
                         window_overlap_factor=45, extra_prefix='joints_worldnorm.pkl')


pd_valid = proxDatasetSkeleton(root_dir='../data_valid/PROXD/', in_frames=in_frames, pred_frames=pred_frames, \
                             output_type='raw_pkls', smplx_model_path=smplx_model_path, frame_jump=6,
                             window_overlap_factor=8, extra_prefix='joints_worldnorm.pkl')


pd_train.sequences = [seq for seq in pd_train.sequences if not any([area in seq[0] for area in val_areas])]
pd_valid.sequences = [seq for seq in pd_valid.sequences if any([area in seq[0] for area in val_areas])]


pdc = DatasetBase(root_dir='../data_valid/recordings/', in_frames=in_frames, pred_frames=pred_frames,
                  search_prefix='Color', extra_prefix='', frame_jump=frame_jump,
                  window_overlap_factor=8)

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
in_skels_all = []
fut_skels_all = []
for i, (idx, in_skels, fut_skels) in (pbar := tqdm.tqdm(enumerate(dataloader_train), total=len(dataloader_train))):
    in_skels_all.append(torch.flatten(in_skels, start_dim=2))
    fut_skels_all.append(torch.flatten(fut_skels, start_dim=2))
in_skels_all = torch.cat(in_skels_all)
fut_skels_all = torch.cat(fut_skels_all)
mean_in_skels_all = in_skels_all.mean(dim=0).unsqueeze(0).detach().clone().to(device)
std_in_skels_all = in_skels_all.std(dim=0).unsqueeze(0).detach().clone().to(device)
mean_fut_skels_all = fut_skels_all.mean(dim=0).unsqueeze(0).detach().clone().to(device)
std_fut_skels_all = fut_skels_all.std(dim=0).unsqueeze(0).detach().clone().to(device)

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
    losses_full_pred = []
    losses_rep = []

    losses_valid = []
    losses_rep_valid = []
    for i, (idx, in_skels, fut_skels) in (pbar := tqdm.tqdm(enumerate(dataloader_train), total=len(dataloader_train))):
        in_skels = in_skels.to(device)
        in_skels_cpy = in_skels.clone()
        fut_skels = fut_skels.to(device)

        in_skels = torch.flatten(in_skels, start_dim=2)
        fut_skels = torch.flatten(fut_skels, start_dim=2)
        # print(in_skels.shape, fut_skels.shape)

        in_skels_norm = (in_skels - mean_in_skels_all) / std_in_skels_all
        fut_skels_norm = (fut_skels - mean_fut_skels_all) / std_fut_skels_all

        pelvis = in_skels_norm[:, 0, :].unsqueeze(1).detach().clone()
        in_skels_norm = in_skels_norm - pelvis

        fut_skels_norm = fut_skels_norm - pelvis
        tgt = torch.cat((in_skels_norm[:, -1, :].unsqueeze(1), fut_skels_norm[:, :-1, :]), dim=1)

        tgt_mask = model.get_tgt_mask(fut_skels_norm.shape[1]).to(device)

        optimizer.zero_grad()

        pred_frames = fut_skels_norm.shape[1]
        batch_len = fut_skels_norm.shape[0]

        pred_skels_norm = model(in_skels_norm, tgt, tgt_mask=tgt_mask)
        pred_skels = ((pred_skels_norm + pelvis) * std_fut_skels_all) + mean_fut_skels_all

        loss = criterion(pred_skels, fut_skels)

        loss.backward()
        if loss.item() < max_loss:
            optimizer.step()

        with torch.no_grad():
            tgt = in_skels_norm[:, -1, :].unsqueeze(1)

            for i in range(fut_skels_norm.shape[1]):
                tgt_mask = model.get_tgt_mask(tgt.size(1)).to(device)
                pred = model(in_skels_norm, tgt, tgt_mask)
                tgt = torch.cat((tgt, pred[:, -1, :].unsqueeze(1)), dim=1)
            pred_skels_norm = tgt[:, 1:, :]
            pred_skels = ((pred_skels_norm + pelvis) * std_fut_skels_all) + mean_fut_skels_all

            losses_full_pred.append(criterion(pred_skels, fut_skels).item())

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
            f"avg last 20 loss: {np.mean(list(filter(lambda x: x < max_loss, losses_full_pred[-20:]))):.4f} ")

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

        # wandb.log({'batch_train_loss': loss.item(),
        #            'batch_train_reploss': loss_rep.item()})
    print(f'end epoch {epoch}: total mean loss: {np.mean(list(filter(lambda x: x < max_loss, losses)))}, '
          f'total full pred loss: {np.mean(list(filter(lambda x: x < max_loss, losses_full_pred)))}, '
          f'mean rep loss: {np.mean(list(filter(lambda x: x < max_loss, losses_rep)))}')



    print('validation loss')
    with torch.no_grad():
        for i, (idx, in_skels, fut_skels) in (
        pbar := tqdm.tqdm(enumerate(dataloader_valid), total=len(dataloader_valid))):
            in_skels = in_skels.to(device)
            in_skels_cpy = in_skels.clone()
            fut_skels = fut_skels.to(device)
            in_skels = torch.flatten(in_skels, start_dim=2)
            fut_skels = torch.flatten(fut_skels, start_dim=2)
            in_skels_norm = (in_skels - mean_in_skels_all) / std_in_skels_all
            fut_skels_norm = (fut_skels - mean_fut_skels_all) / std_fut_skels_all

            pelvis = in_skels_norm[:, 0, :].unsqueeze(1)
            in_skels_norm = in_skels_norm - pelvis

            fut_skels_norm = fut_skels_norm - pelvis
            tgt = in_skels_norm[:, -1, :].unsqueeze(1)

            for i in range(fut_skels_norm.shape[1]):
                tgt_mask = model.get_tgt_mask(tgt.size(1)).to(device)
                pred = model(in_skels_norm, tgt, tgt_mask)
                tgt = torch.cat((tgt, pred[:, -1, :].unsqueeze(1)), dim=1)
            pred_skels_norm = tgt[:, 1:, :]
            pred_skels = ((pred_skels_norm + pelvis) * std_fut_skels_all) + mean_fut_skels_all

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
            # wandb.log({'batch_val_loss': loss.item(),
            #            'batch_val_reploss': loss_rep.item()})
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
            }, os.path.join(save_folder, 'transformer_best_model' + name + '.pt'))
