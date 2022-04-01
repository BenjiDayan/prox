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

        self.input_fc = nn.Linear(input_size, 512)

        self.GRUcell_list = [torch.nn.GRUCell(input_size=512, hidden_size=hidden_size)]
        for i in range(1, n_layers):
            self.GRUcell_list.append(torch.nn.GRUCell(input_size=hidden_size, hidden_size=hidden_size))
        self.GRUcell_list = nn.ModuleList(self.GRUcell_list)

        self.fc1 = nn.Linear(hidden_size, input_size)

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


