import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class PoseGRU_inputFC(nn.Module):
    def __init__(self, batch_size=10, input_size=3 * 21, hidden_size=512, n_layers=2, n_joint=21):
        super(PoseGRU_inputFC, self).__init__()
        self.n_joint = n_joint
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.input_fc = nn.Linear(input_size, 512).to(device)

        self.GRUcell_list = [torch.nn.GRUCell(input_size=512, hidden_size=hidden_size)]
        for i in range(1, n_layers):
            self.GRUcell_list.append(torch.nn.GRUCell(input_size=hidden_size, hidden_size=hidden_size))
        self.GRUcell_list = nn.ModuleList(self.GRUcell_list).to(device)

        self.fc1 = nn.Linear(hidden_size, input_size).to(device)

    def init_hidden(self):
        self.hidden = []  # if n_lalyer=2:   [h, h]   all zeros
        for i in range(self.n_layers):
            self.hidden.append(Variable(torch.zeros(self.batch_size, self.hidden_size)).to(device))

    def forward(self, input):
        # input: pose3d [bs, 3, 21]
        input = input.view(-1, 3 * self.n_joint)  # [bs, 63]
        input_feat = self.input_fc(input)  # [bs, 512]

        for i in range(self.n_layers):
            if i == 0:
                self.hidden[i] = self.GRUcell_list[i](input_feat, self.hidden[i])  # self.hidden[i]: h
            else:
                self.hidden[i] = self.GRUcell_list[i](cur_state, self.hidden[i])  # self.hidden[i]: h
            cur_state = self.hidden[i]  # input to next layer: output h of current layer  [bs, hidden_size]

        output = self.fc1(cur_state)  # [bs, input_size]
        output += input
        output = output.view(-1, 3, self.n_joint)

        return cur_state, output  # gru state [bs, hidden_size], output [bs, 3, 21]
        

class PoseGRU_inputFC2(nn.Module):
    def __init__(self, input_size=(21, 3), hidden_size=512, n_layers=2):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.input_fc = nn.Linear(np.product(input_size), 512).to(device)

        self.GRUcell_list = [torch.nn.GRUCell(input_size=512, hidden_size=hidden_size)]
        for i in range(1, n_layers):
            self.GRUcell_list.append(torch.nn.GRUCell(input_size=hidden_size, hidden_size=hidden_size))
        self.GRUcell_list = nn.ModuleList(self.GRUcell_list).to(device)

        self.fc1 = nn.Linear(hidden_size, np.product(input_size)).to(device)

    def init_hidden(self, batch_size):
        self.hidden = []  # if n_lalyer=2:   [h, h]   all zeros
        for i in range(self.n_layers):
            self.hidden.append(Variable(torch.zeros(batch_size, self.hidden_size)).to(device))

    def forward(self, input, hidden=None):
        # input: pose3d [bs, 21, 3]
        bs = input.shape[0]
        input = input.view(bs, np.product(self.input_size))  # [bs, 63]
        input_feat = self.input_fc(input)  # [bs, 512]
        if hidden is None:
            self.init_hidden(bs)

        for i in range(self.n_layers):
            if i == 0:
                self.hidden[i] = self.GRUcell_list[i](input_feat, self.hidden[i])  # self.hidden[i]: h
            else:
                self.hidden[i] = self.GRUcell_list[i](cur_state, self.hidden[i])  # self.hidden[i]: h
            cur_state = self.hidden[i]  # input to next layer: output h of current layer  [bs, hidden_size]

        output = self.fc1(cur_state)  # [bs, np.product(input_size)]
        # residual
        output += input
        output = output.view((bs,) + self.input_size)
        return cur_state, output  # gru state [bs, hidden_size], output [bs, 3, 21]

    def forward_prediction(self, in_seq: torch.Tensor, out_seq_len, all=False):
        # in_seq: pose3d [bs, in_seq_len, 21, 3]
        in_seq_len = in_seq.shape[1]
        predictions = []
        for time_idx in range(in_seq_len + out_seq_len-1):
            if time_idx == 0:
                cur_state, output = self.forward(in_seq[:, time_idx])
            if time_idx < in_seq_len:
                cur_state, output = self.forward(in_seq[:, time_idx], hidden=cur_state)
            else:
                cur_state, output = self.forward(output, hidden=cur_state)

            if not all and time_idx >= in_seq_len - 1:
                predictions.append(output)
            else:
                predictions.append(output)

        pred_skels = torch.cat([tens.view((1,) + tens.shape) for tens in predictions])
        pred_skels = pred_skels.transpose(0, 1)

        return cur_state, pred_skels

    def forward_prediction_guided(self, in_seq, out_seq):
        # input: pose3d [bs, in_seq_len, 21, 3]
        in_seq_len = in_seq.shape[1]
        out_seq_len = out_seq.shape[1]
        predictions = []
        for time_idx in range(in_seq_len + out_seq_len - 1):
            if time_idx == 0:
                cur_state, output = self.forward(in_seq[:, time_idx])
            if time_idx < in_seq_len:
                cur_state, output = self.forward(in_seq[:, time_idx], hidden=cur_state)
            else:
                cur_state, output = self.forward(out_seq[:, time_idx-in_seq_len], hidden=cur_state)

            if time_idx >= in_seq_len - 1:
                predictions.append(output)

        pred_skels = torch.cat([tens.view((1,) + tens.shape) for tens in predictions])
        pred_skels = pred_skels.transpose(0, 1)

        return cur_state, pred_skels

# class PoseGRU_inputFC3(nn.Module):
#     def __init__(self, batch_size=10, input_size=(3, 21), hidden_size=512, n_layers=2):
#         super().__init__()
#         self.input_size = input_size
#         self.batch_size = batch_size
#         self.hidden_size = hidden_size
#         self.n_layers = n_layers

#         self.input_fc = nn.Linear(input_size, 512)

#         self.GRUcell_list = [torch.nn.GRUCell(input_size=512, hidden_size=hidden_size)]
#         for i in range(1, n_layers):
#             self.GRUcell_list.append(torch.nn.GRUCell(input_size=hidden_size, hidden_size=hidden_size))
#         self.GRUcell_list = nn.ModuleList(self.GRUcell_list)

#         self.fc1 = nn.Linear(hidden_size, input_size)

#     def init_hidden(self, batch_size):
#         self.hidden = []  # if n_lalyer=2:   [h, h]   all zeros
#         for i in range(self.n_layers):
#             self.hidden.append(Variable(torch.zeros(batch_size, self.hidden_size)).to(device))

#     def forward(self, input):
#         # input: pose3d [bs, seq_len, 21, 3]
#         input = input.view(input.shape[0], input.shape[1], np.product(input_size))  # [bs, 63]
#         self.init_hidden(input.shape[0])
#         input_feat = self.input_fc(input)  # [bs, 512]

#         for i in range(self.n_layers):
#             if i == 0:
#                 self.hidden[i] = self.GRUcell_list[i](input_feat, self.hidden[i])  # self.hidden[i]: h
#             else:
#                 self.hidden[i] = self.GRUcell_list[i](cur_state, self.hidden[i])  # self.hidden[i]: h
#             cur_state = self.hidden[i]  # input to next layer: output h of current layer  [bs, hidden_size]

#         output = self.fc1(cur_state)  # [bs, input_size]
#         output += input
#         output = output.view(-1, 3, self.n_joint)

#         return cur_state, output  # gru state [bs, hidden_size], output [bs, 3, 21]

