import argparse
import torch
from torch.utils import data
import time
from tqdm import tqdm
from loader.train_dataloader import TrainLoader
from tensorboardX import SummaryWriter
import smplx
import torch.optim as optim
import itertools
import h5py
import sys
from utils import *
from models.cnn_lstm_unet import *
from models.pose_gru import *
import random
import os

parser = argparse.ArgumentParser()
parser.add_argument('--save_dir', type=str, default='runs_try')
parser.add_argument('--batch_size', type=int, default=5)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--num_epoch', type=int, default=100000)

parser.add_argument("--log_step", default=200, type=int)
parser.add_argument("--save_step", default=200, type=int)
parser.add_argument('--load_to_memory', action='store_true')





args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def loop_function(prev):
    return prev

def train(writer, logger):
    # data_root = '/local/home/szhang/GTA-1M'
    data_root = '/cluster/scratch/szhang/GTA-1M'
    train_dataset = TrainLoader(split='train', h=256, w=448, read_memory=args.load_to_memory)
    train_dataset.load_data(data_root=data_root)
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                                                   shuffle=True, num_workers=2, drop_last=True)

    test_dataset = TrainLoader(split='test', h=256, w=448, read_memory=args.load_to_memory)
    test_dataset.load_data(data_root=data_root)
    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.batch_size,
                                                  shuffle=False, num_workers=2, drop_last=True)

    # GRU = PoseGRU(batch_size=args.batch_size, input_size=3*21, hidden_size=1024, n_layers=1, n_joint=21).to(device)
    GRU = PoseGRU_inputFC(batch_size=args.batch_size, input_size=3 * 21, hidden_size=1024, n_layers=1, n_joint=21).to(device)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                                  itertools.chain(GRU.parameters(),)),
                           lr=args.lr)

    ####################### train #########################################
    total_steps = 0
    for epoch in range(args.num_epoch):
        for step, data in tqdm(enumerate(train_dataloader)):
            GRU.train()

            total_steps += 1
            # img: [bs, 3, h, w]
            # depth: [bs, 1, h, w]
            # bps_seq: [bs, 15, 1, h, w]
            # pose3d_seq: [bs, 15, 3, 21]
            [_, _, _, _, pose3d_seq, _] = [item.to(device) for item in data[0:-1]]

            optimizer.zero_grad()

            GRU.init_hidden()

            loss_pose3d_rec_past, loss_pose3d_rec_future = 0, 0
            ############# encode img
            # img_feat = encoderImg(torch.cat([img, depth], dim=1))  # [bs, 512, 4, 7]
            for t in range(1, 15):
                ############# encode
                if t <= 5:
                    gru_state, pose3d_rec_t = GRU(pose3d_seq[:, t-1,])
                else:
                    prev_pred = loop_function(pose3d_rec_t)
                    prev_pred = prev_pred.detach()
                    gru_state, pose3d_rec_t = GRU(prev_pred)
                    # gru_state, pose3d_rec_t = GRU(pose3d_rec_t)
                ############## compute loss
                if t <= 4:
                    loss_pose3d_rec_past += F.l1_loss(pose3d_seq[:, t,], pose3d_rec_t)
                else:
                    loss_pose3d_rec_future += F.l1_loss(pose3d_seq[:, t,], pose3d_rec_t)
            loss_pose3d_rec_past = loss_pose3d_rec_past / 4
            loss_pose3d_rec_future = loss_pose3d_rec_future / 10


            loss = loss_pose3d_rec_future + loss_pose3d_rec_past
            loss.backward()
            optimizer.step()

            if total_steps % args.log_step == 0:
                writer.add_scalar('train/loss_pose3d_rec_past', loss_pose3d_rec_past.item(), total_steps)
                print_str = 'Step {:d}/ Epoch {:d}]  loss_pose3d_rec_past: {:.6f}'. \
                    format(step, epoch, loss_pose3d_rec_past.item())
                logger.info(print_str)
                print(print_str)

                writer.add_scalar('train/loss_pose3d_rec_future', loss_pose3d_rec_future.item(), total_steps)
                print_str = 'Step {:d}/ Epoch {:d}]  loss_pose3d_rec_future: {:.6f}'. \
                    format(step, epoch, loss_pose3d_rec_future.item())
                logger.info(print_str)
                print(print_str)



            ################## test loss #################################
            if total_steps % args.log_step == 0:
                loss_pose3d_rec_test_past = 0
                loss_pose3d_rec_test_future = 0
                with torch.no_grad():
                    for test_step, data in tqdm(enumerate(test_dataloader)):
                        GRU.eval()

                        [_, _, _, _, pose3d_seq, _] = [item.to(device) for item in data[0:-1]]

                        GRU.init_hidden()

                        ############# encode img
                        # img_feat = encoderImg(torch.cat([img, depth], dim=1))  # [bs, 512, 4, 7]
                        for t in range(1, 15):
                            ############# encode
                            if t <= 5:
                                gru_state, pose3d_rec_t = GRU(pose3d_seq[:, t - 1, ])
                            else:
                                gru_state, pose3d_rec_t = GRU(pose3d_rec_t)
                            ############## compute loss
                            if t <= 4:
                                loss_pose3d_rec_test_past += F.l1_loss(pose3d_seq[:, t,], pose3d_rec_t).item()
                            else:
                                loss_pose3d_rec_test_future += F.l1_loss(pose3d_seq[:, t,], pose3d_rec_t).item()

                    loss_pose3d_rec_test_past = loss_pose3d_rec_test_past / test_step / 4
                    loss_pose3d_rec_test_future = loss_pose3d_rec_test_future / test_step / 10


                writer.add_scalar('test/loss_pose3d_rec_test_past', loss_pose3d_rec_test_past, total_steps)
                print_str = 'Step {:d}/ Epoch {:d}]  loss_pose3d_rec_test_past: {:.6f}'. \
                    format(step, epoch, loss_pose3d_rec_test_past)
                logger.info(print_str)
                print(print_str)

                writer.add_scalar('test/loss_pose3d_rec_test_future', loss_pose3d_rec_test_future, total_steps)
                print_str = 'Step {:d}/ Epoch {:d}]  loss_pose3d_rec_test_future: {:.6f}'. \
                    format(step, epoch, loss_pose3d_rec_test_future)
                logger.info(print_str)
                print(print_str)

            if total_steps % args.save_step == 0:
                save_path = os.path.join(writer.file_writer.get_logdir(), "GRU.pkl")
                torch.save(GRU.state_dict(), save_path)
                print('[*] last model saved\n')
                logger.info('[*] last model saved\n')




if __name__ == '__main__':
    run_id = random.randint(1, 100000)
    logdir = os.path.join(args.save_dir, str(run_id))  # create new path
    writer = SummaryWriter(log_dir=logdir)
    print('RUNDIR: {}'.format(logdir))
    sys.stdout.flush()

    logger = get_logger(logdir)
    logger.info('Let the games begin')  # write in log file
    save_config(logdir, args)
    train(writer, logger)

























