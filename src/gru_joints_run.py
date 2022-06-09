import numpy as np # for data manipulation
print('numpy: %s' % np.__version__) # print version
import math # to help with data reshaping of the data

import numpy as np
import torch
#import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

# from sklearn.model_selection import train_test_split
import tqdm
import matplotlib.pyplot as plt
import logging


from pose_gru import PoseGRU_inputFC2
from benji_prox_dataloader import *


# root_dir = "D:/prox_data/joint_locations/"
root_dir = "/cluster/scratch/bdayan/prox_data/joint_locations/"
batch_size = 15

pd = proxDatasetJoints(root_dir)
dataloader = DataLoader(pd, batch_size=batch_size,
                        shuffle=True, num_workers=0, collate_fn=my_collate)



criterion = nn.MSELoss()
learning_rate=0.0001
losses = []
losses_rep = []

# rnn = MockupModel(input_size=21*3, hidden_size=50, num_layers=2, output_size=(pred_frames, 21, 3), seq_len=in_frames)
gru = PoseGRU_inputFC2(input_size=(25,3))

optimizer = torch.optim.Adam(gru.parameters(), lr=learning_rate)

n_iter = 10  # TODO increase
i = 0

save_every=100

save_path='model_saves/model_epoch{epoch}_bn{batchnum}.pt'
writer = SummaryWriter()

idx_counter = 0

for epoch in range(n_iter):
    for i, (indices, in_skels, fut_skels) in (pbar := tqdm.tqdm(enumerate(dataloader))):
        # if i % 300 == 298:  # TODO remove
        #     break
        optimizer.zero_grad()
        
        pred_frames = fut_skels.shape[1]
        
        cur_state, pred_skels = gru.forward_prediction(in_skels, pred_len=pred_frames)
        loss = criterion(pred_skels, fut_skels)
        loss.backward()
        optimizer.step() 

        rep_pred = in_skels[:, -1, :, :]
        rep_pred = rep_pred.tile(pred_frames, 1, 1, 1).transpose(0, 1)
        loss_rep = criterion(rep_pred, fut_skels)
        losses_rep.append(loss_rep.item())
        
        losses.append(loss.item())
        pbar.set_description(f"avg last 20 loss: {np.mean(losses[-20:]):.4f} avg last 200-100: {np.mean(losses[-200:-100]):.4f}")

        writer.add_scalar('Loss', losses[-1], idx_counter)
        writer.add_scalar('Loss_rep', losses_rep[-1], idx_counter)
        # if i % save_every == (save_every-1):
        #     torch.save({
        #     'epoch': epoch,
        #     'batch_num': i,
        #     'model_state_dict': gru.state_dict(),
        #     'optimizer_state_dict': optimizer.state_dict(),
        #     'loss': loss,
        #     }, save_path.format(epoch=epoch, batchnum=i))
            
        idx_counter += 1
    print(f'end epoch {epoch}: total mean loss: {np.mean(losses)}')
    torch.save({
            'epoch': epoch,
            'batch_num': i,
            'model_state_dict': gru.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, save_path.format(epoch=epoch, batchnum=i))

plt.plot(losses)
print(losses[-4:])