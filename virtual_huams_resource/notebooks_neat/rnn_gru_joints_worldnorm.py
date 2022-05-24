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

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


import sys
sys.path.append('../')
sys.path.append('../../src')

from pose_gru import PoseGRU_inputFC2
from benji_prox_dataloader import *

print(f'cuda availability: {torch.cuda.is_available()}')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'device: {device}')

name = "GRU_joints_15_30_3fps_24_05_1854"

import wandb
_ = wandb.init(settings=wandb.Settings(start_method="fork"), project="vh-human-motion-pred", entity="benjidayan", name=name)


root_dir = "/cluster/scratch/bdayan/prox_data"
smplx_model_path='/cluster/home/bdayan/prox/prox/models_smplx_v1_1/models/'

batch_size = 15
in_frames=15
pred_frames=30
frame_jump=10
window_overlap_factor=8
lr=0.0001
n_iter = 200
save_every=40
num_workers=0
max_loss = 5. # This is dangerous but stops ridiculous updates?
bsub_command = 'bsub -n 20 -R "rusage[mem=16384,ngpus_excl_p=1]" python rnn_gru_joints_worldnorm.py'

save_folder = 'saves'
os.makedirs(save_folder, exist_ok=True)

save_path=os.path.join(save_folder, name + '_epoch{epoch}_bn{batchnum}.pt')
save_path.format(epoch=3, batchnum=5)

pd = proxDatasetSkeleton(root_dir=root_dir + '/PROXD', in_frames=in_frames, pred_frames=pred_frames, \
                       output_type='raw_pkls', smplx_model_path=smplx_model_path, frame_jump=frame_jump, window_overlap_factor=window_overlap_factor, extra_prefix='joints_worldnorm.pkl')


def my_collate2(batch):
    # I think these are still np.arrays, will become tensors later
    
    batch = list(filter(
        # check that they exist and don't have nans??
        lambda triple: (triple[1] is not None) and (triple[2] is not None),
        batch
    ))
    try:
        batch = [(triple[0], torch.stack(triple[1][1]).squeeze(), torch.stack(triple[2][1]).squeeze()) for triple in batch]
    
        batch = list(filter(
            lambda triple: (not torch.any(torch.isnan(triple[1]))) and (not torch.any(torch.isnan(triple[2]))), batch
        ))
        return default_collate(batch)
        
    except Exception as e:
        print(f'batch: {[(triple[0], triple[1].shape, triple[2].shape) for triple in batch]}')
        print(e, e.args)
        raise e
        
dataloader = DataLoader(pd, batch_size=batch_size,
                        shuffle=True, num_workers=0, collate_fn=my_collate2)



criterion = nn.MSELoss()
losses = []
losses_rep = []

gru = PoseGRU_inputFC2(input_size=(25,3)).to(device)

optimizer = torch.optim.Adam(gru.parameters(), lr=lr)


writer = SummaryWriter()

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
    "bsub_command": bsub_command
}

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
        if loss.item() < max_loss:
            optimizer.step() 

        rep_pred = in_skels[:, -1, :, :]
        a = rep_pred.detach().cpu().numpy()
        
        a = np.tile(a, (fut_skels.shape[1], 1, 1, 1))
        rep_pred = torch.Tensor(a).transpose(0, 1).to(device)

        loss_rep = criterion(rep_pred, fut_skels)
        losses_rep.append(loss_rep.item())
        
        losses.append(loss.item())
        
        wandb.log({'MSEloss': loss, 'rep_pred_MSEloss': loss_rep})
        
        pbar.set_description(f"avg last 20 loss: {np.mean(losses[-20:]):.4f} avg last 200-100: {np.mean(losses[-200:-100]):.4f}")

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

plt.plot(losses)
print(losses[-4:])