import os
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import tensorflow as tf
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math
import random


class InnerProductDecoder(torch.nn.Module):

    def forward(self, z, edge_index, sigmoid = True):
        value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        return torch.sigmoid(value) if sigmoid else value




class Attention(nn.Module):
    def __init__(self, hidden_size, time_steps):
        super(Attention, self).__init__()
        self.lstm_units = hidden_size
        self.time_steps = time_steps


    def forward(self, inputs):
        a = inputs.permute(1,0)
        a_probs = torch.nn.functional.softmax(a, dim=-1)
        a_probs = torch.transpose(a_probs, 0, 1)
        output_attention_mul = inputs * a_probs

        return output_attention_mul




class GraphTrans_Encoder(torch.nn.Module):
    def __init__(self, num_features, hidden_dim, nhead = 4, num_layers=1,dropout=0.2, is_training=False):
        super(GraphTrans_Encoder, self).__init__()
        self.linner = nn.Linear(num_features, hidden_dim)
        self.linner1 = nn.Linear(hidden_dim, hidden_dim)
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=int(hidden_dim/2), num_layers=2, bidirectional=True)
        self.is_training = is_training
        self.attention1 = Attention(hidden_size=int(hidden_dim/2), time_steps=100)
        self.dropout = 0.5


    def forward(self, x, edge_index):
        x = self.linner(x)
        x = x.relu()

        x1, _ = self.lstm(x)
        x1 = x1.relu()
        x1 = x1 + self.attention1(x1)

        x1 = self.conv2(x1, edge_index)
        x1 = x1.relu()

        x = x + x1

        x2, _ = self.lstm(x)
        x2 = x2.relu()
        x2 = x2+self.attention1(x2)
        x2 = self.conv2(x2, edge_index)
        x = F.dropout(x2, p=self.dropout, training=self.is_training)

        return x
    

class CellTEncoder(torch.nn.Module):
    def __init__(self, GraphTrans_Encoder,is_training=False):
        super(CellTEncoder, self).__init__()
        self.encoder_c = GraphTrans_Encoder
        self.is_training = is_training


    def forward(self, x_c, edge_index_c):
        Z_c = self.encoder_c(x_c, edge_index_c) 
        Z = Z_c
        return Z, Z_c






