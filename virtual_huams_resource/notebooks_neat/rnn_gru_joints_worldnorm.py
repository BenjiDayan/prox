import os
os.environ['PYOPENGL_PLATFORM']='osmesa'


import numpy as np # for data manipulation
print('numpy: %s' % np.__version__) # print version
np.random.seed(0)
import math # to help with data reshaping of the data

import numpy as np
import torch
torch.manual_seed(0)

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter


import tqdm
import matplotlib.pyplot as plt
import logging
import wandb
import json

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


import sys
sys.path.append('../')
sys.path.append('../../src')

from pose_gru import PoseGRU_inputFC2
from benji_prox_dataloader import *
from visualisation import predict_and_visualise

print(f'cuda availability: {torch.cuda.is_available()}')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'device: {device}')

name = "GRU_joints_15_30_3fps_2layers256_27_05_1435"

root_dir = "/cluster/scratch/bdayan/prox_data"
smplx_model_path='/cluster/home/bdayan/prox/prox/models_smplx_v1_1/models/'

batch_size = 15
in_frames=15
pred_frames=30
frame_jump=10
window_overlap_factor=8
lr=0.0001
n_layers=2
n_iter = 300
save_every=40
num_workers=4
hidden_size=256
max_loss = 5. # This is dangerous but stops ridiculous updates?
bsub_command = 'bsub -n 6 -R "rusage[mem=2048,ngpus_excl_p=1]" python rnn_gru_joints_worldnorm.py' 
# 'bsub -n 8 -R "rusage[mem=16384,ngpus_excl_p=1]" python rnn_gru_joints_worldnorm.py'

wandb.config = {
    "learning_rate": lr,
    "epochs": n_iter,
    "batch_size": batch_size,
    "in_frames": in_frames,
    "pred_frames": pred_frames,
    "frame_jump": frame_jump,
    "window_overlap_factor": window_overlap_factor,
    "max_loss": max_loss,
    "num_workers": num_workers,
    "n_layers": n_layers,
    "hidden_size": hidden_size,
    "bsub_command": bsub_command
}

save_folder = 'saves'
os.makedirs(save_folder, exist_ok=True)

save_path=os.path.join(save_folder, name + '_epoch{epoch}_bn{batchnum}.pt')
save_path.format(epoch=3, batchnum=5)

pd = proxDatasetSkeleton(root_dir=root_dir + '/PROXD', in_frames=in_frames, pred_frames=pred_frames, \
                       output_type='raw_pkls', smplx_model_path=smplx_model_path, frame_jump=frame_jump, window_overlap_factor=window_overlap_factor, extra_prefix='joints_worldnorm.pkl')

val_areas =['BasementSittingBooth', 'N3OpenArea']

pd = proxDatasetSkeleton(root_dir=root_dir + '/PROXD', in_frames=in_frames, pred_frames=pred_frames, \
                       output_type='raw_pkls', smplx_model_path=smplx_model_path, frame_jump=frame_jump, window_overlap_factor=window_overlap_factor, extra_prefix='joints_worldnorm.pkl')

pd_val = proxDatasetSkeleton(root_dir=root_dir + '/PROXD', in_frames=in_frames, pred_frames=pred_frames, \
                       output_type='raw_pkls', smplx_model_path=smplx_model_path, frame_jump=frame_jump, window_overlap_factor=window_overlap_factor, extra_prefix='joints_worldnorm.pkl')

pd.sequences = [seq for seq in pd.sequences if not any([area in seq[0] for area in val_areas])]
pd_val.sequences = [seq for seq in pd_val.sequences if any([area in seq[0] for area in val_areas])]

pdc = DatasetBase(root_dir=root_dir + '/recordings', in_frames=in_frames, pred_frames=pred_frames,
                                             search_prefix='Color', extra_prefix='', frame_jump=frame_jump, window_overlap_factor=window_overlap_factor)

pdc.align(pd_val)
pd_val.align(pdc)
print('length pdc, pd_val:')
print(len(pdc))
print(len(pd_val))


def my_collate2(batch):
    # I think these are still np.arrays, will become tensors later
    
    batch = list(filter(
        # check that they exist and don't have nans??
        lambda triple: (triple[1] is not None) and (triple[2] is not None),
        batch
    ))
    try:
        with torch.no_grad():  # Somehow this is necessary when num_workers > 0? Input data tensor don't want grad anyway
            batch = [(triple[0], torch.stack(triple[1][1]).squeeze(), torch.stack(triple[2][1]).squeeze()) for triple in batch]
    
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
        
        
        
dataloader = DataLoader(pd, batch_size=batch_size,
                        shuffle=True, num_workers=num_workers, pin_memory=True, collate_fn=my_collate2)
dataloader_val = DataLoader(pd_val, batch_size=batch_size,
                        shuffle=True, num_workers=num_workers, pin_memory=True, collate_fn=my_collate2)



criterion = nn.MSELoss()
losses = []
losses_rep = []

gru = PoseGRU_inputFC2(input_size=(25,3), n_layers=n_layers, hidden_size=hidden_size).to(device)

optimizer = torch.optim.Adam(gru.parameters(), lr=lr)


writer = SummaryWriter()


# I think it's important to do this after defining config? annoying though.
_ = wandb.init(settings=wandb.Settings(start_method="fork"), project="rnn", entity="vh-motion-pred", name=name, config=dict(wandb.config))

# Actually this doesn't work :( write manuallly
with open('config.json', 'w') as file:
    file.write(json.dumps(dict(wandb.config)))
    
wandb.save('config.json')
# os.remove('config.json')

idx_counter = 0
last_fn = None
for epoch in range(n_iter):
    for i, (idx, in_skels, fut_skels) in (pbar := tqdm.tqdm(enumerate(dataloader), total=len(dataloader))):
        in_skels = in_skels.to(device)
        # print(f'in_skels device: {in_skels.device}')
        fut_skels = fut_skels.to(device)
        
        pelvis = in_skels[:, 0, 0, :].unsqueeze(1).unsqueeze(1)
        in_skels = in_skels - pelvis
        fut_skels = fut_skels - pelvis
        
        optimizer.zero_grad()
        
        pred_frames = fut_skels.shape[1]
        batch_len = fut_skels.shape[0]
        # print(f'batch_len: {batch_len}')  # maybe something's wrong but I do get about avg 13 batchlen not 15 :( crummy files?
        
        cur_state, pred_skels = gru.forward_prediction(in_skels, pred_len=pred_frames)
        loss = criterion(pred_skels, fut_skels)
        loss.backward()
        # if loss.item() < max_loss:
        optimizer.step() 

        rep_pred = in_skels[:, -1, :, :]
        a = rep_pred.detach().cpu().numpy()
        
        a = np.tile(a, (fut_skels.shape[1], 1, 1, 1))
        rep_pred = torch.Tensor(a).transpose(0, 1).to(device)

        loss_rep = criterion(rep_pred, fut_skels)
        losses_rep.append(loss_rep.item())
        
        losses.append(loss.item())
        
        wandb.log({'MSEloss': loss, 'rep_pred_MSEloss': loss_rep})
        
        pbar.set_description(f"avg last 20 loss: {np.mean(losses[-20:]):.4f}")

        writer.add_scalar('Loss', losses[-1], idx_counter)
        writer.add_scalar('Loss_rep', losses_rep[-1], idx_counter)
        if i % save_every == (save_every-1):
            fn = save_path.format(epoch=epoch, batchnum=i)
            torch.save({
            'epoch': epoch,
            'batch_num': i,
            'model_state_dict': gru.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, fn)
            wandb.save(fn)       
            if last_fn:
                os.remove(last_fn)
            last_fn = fn
            
        idx_counter += 1
    print(f'end epoch {epoch}: total mean loss: {np.mean(losses)}')
    
    epoch_val_losses = []
    epoch_val_replosses = []
    
    # visualisation on validation
    if epoch % 5 == 5-1:
        for i in range(3):
            idx = np.random.randint(pd_val.bounds[-1])
            try:
                (_, (_, in_skels), (_, fut_skels)) = pd_val.__getitem__(idx)
                _, in_frames_fns, _, pred_frames_fns = pdc.__getitem__(idx)

                in_imgs = [np.array(cv2.imread(fn)) for fn in in_frames_fns]
                fut_imgs = [np.array(cv2.imread(fn)) for fn in pred_frames_fns]
            except Exception as e:  # some skel fn None so get idx None, None. Or image files unreadable etc.
                continue
                
            if in_skels is not None and fut_skels is not None:
                in_skels_world = torch.cat(in_skels)
                fut_skels_world = torch.cat(fut_skels)
            if torch.any(torch.isnan(in_skels_world)) or  torch.any(torch.isnan(fut_skels_world)):
                continue

            start, seq_idx = get_start_idx(idx, pd_val.bounds, pd_val.start_jump)
            scene_name = pd_val.sequences[seq_idx][0]
            scene_name = scene_name[:scene_name.index('_')]
            with open(f'{root_dir}/cam2world/{scene_name}.json') as file:
                cam2world = np.array(json.load(file))
                cam2world = torch.from_numpy(cam2world).float()

            images = in_imgs + fut_imgs

            output_images = predict_and_visualise(gru, in_skels_world, fut_skels_world, images, cam2world)
            images_down = [cv2.resize((img*255).astype(np.uint8), dsize=(int(img.shape[1]/5), int(img.shape[0]/5))) for img in output_images]
            wandb.log({f'val_image_seq{i}': [wandb.Image(img) for img in images_down]})
                                
        
    
    for i, (idx, in_skels, fut_skels) in (pbar := tqdm.tqdm(enumerate(dataloader_val), total=len(dataloader_val))):
        in_skels = in_skels.to(device)
        fut_skels = fut_skels.to(device)
        
        pelvis = in_skels[:, 0, 0, :].unsqueeze(1).unsqueeze(1)
        in_skels = in_skels - pelvis
        fut_skels = fut_skels - pelvis

        
        pred_frames = fut_skels.shape[1]
        batch_len = fut_skels.shape[0]
        
        cur_state, pred_skels = gru.forward_prediction(in_skels, pred_len=pred_frames)
        
        loss = criterion(pred_skels, fut_skels)

        rep_pred = in_skels[:, -1, :, :]
        a = rep_pred.detach().cpu().numpy()
        a = np.tile(a, (fut_skels.shape[1], 1, 1, 1))
        rep_pred = torch.Tensor(a).transpose(0, 1).to(device)
        loss_rep = criterion(rep_pred, fut_skels)
        
        epoch_val_replosses.append(loss_rep.item())       
        epoch_val_losses.append(loss.item())
        
        pbar.set_description(f"avg val loss: {np.mean(epoch_val_losses):.4f}")
        
    epoch_val_loss = np.mean(list(filter(lambda loss: loss < max_loss, epoch_val_losses)))
    epoch_val_reploss = np.mean(list(filter(lambda loss: loss < max_loss, epoch_val_replosses)))
    wandb.log({'epoch_val_loss': epoch_val_loss, 'epoch_val_reploss': epoch_val_reploss})
    

plt.plot(losses)
print(losses[-4:])
