import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from dataset_loader import *
import argparse
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from transformers import BertModel, BertTokenizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size*num_layers, 2)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, lengths):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Pack the padded sequences
        packed_input = rnn_utils.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        input, (h_0, c_0) = self.lstm(packed_input, (h0, c0))

        # Reshape 
        h_0 = h_0.view(-1, self.hidden_size*self.num_layers)

        # Unpack the packed sequence
        unpacked_out, _ = rnn_utils.pad_packed_sequence(input, batch_first=True)

        # Take the output of the last time step
        last_output = unpacked_out[torch.arange(len(unpacked_out)), lengths - 1]

        out = self.dropout(self.relu(self.fc(h_0)))
        out = self.sigmoid(out)

        return out.squeeze()


class LayerNormGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gru_cell = nn.GRUCell(input_size, hidden_size)
        self.ln_ih = nn.LayerNorm(input_size)
        self.ln_hh = nn.LayerNorm(hidden_size)
        self.ln_ho = nn.LayerNorm(hidden_size)

    def forward(self, x, h):
        x = self.ln_ih(x)
        h = self.ln_hh(h)
        h_next = self.gru_cell(x, h)
        h_next = self.ln_ho(h_next)
        return h_next


class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, lengths):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Pack the padded sequences
        packed_input = rnn_utils.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        out, _ = self.gru(packed_input, h0)

        # Unpack the packed sequence
        unpacked_out, _ = rnn_utils.pad_packed_sequence(out, batch_first=True)

        # Take the output of the last time step
        last_output = unpacked_out[torch.arange(len(unpacked_out)), lengths - 1]

        out = self.fc(last_output)
        out = self.sigmoid(out)

        return out.squeeze()
    
class GRUModel_LN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(GRUModel_LN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Define a LayerNormGRUCell for each layer
        self.cells = nn.ModuleList([
            LayerNormGRUCell(input_size if i==0 else hidden_size, hidden_size) for i in range(num_layers)
        ])
        self.fc = nn.Linear(hidden_size, 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, lengths):
        batch_size, seq_len, _ = x.size()
        h = torch.zeros(batch_size, self.hidden_size).to(x.device)
        out = []

        # Unpack the sequence and apply the GRU cells
        for t in range(seq_len):
            x_t = x[:, t, :]
            for i in range(self.num_layers):
                h = self.cells[i](x_t, h)
                x_t = h
            out.append(h.unsqueeze(1))

        out = torch.cat(out, dim=1)

        # Take the output of the last time step
        last_output = out[torch.arange(len(out)), lengths - 1]

        out = self.fc(last_output)
        out = self.sigmoid(out)

        return out.squeeze()

class ModifiedLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, cnn_out_channels=32, cnn_kernel_size=3, pool_kernel_size=2, dropout_prob=0.2):
        super(ModifiedLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # CNN layer
        self.cnn = nn.Conv1d(input_size, cnn_out_channels, cnn_kernel_size)
        self.maxpool = nn.MaxPool1d(pool_kernel_size)
        self.dropout_cnn = nn.Dropout(dropout_prob)

        self.lstm = nn.LSTM(cnn_out_channels, hidden_size, num_layers, batch_first=True)
        self.dropout_lstm = nn.Dropout(dropout_prob)

        self.fc = nn.Linear(hidden_size, 2)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x, lengths):
        x = x.permute(0, 2, 1)  # Reshape input for Conv1d

        # CNN layer
        x = self.dropout_cnn(self.relu(self.cnn(x)))
        x = self.maxpool(x)
        x = x.permute(0, 2, 1)

        # LSTM layer
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        output, (h_n, c_n) = self.lstm(x, (h0, c0))
        x = self.dropout_lstm(h_n[-1])

        # Fully connected layer
        x = self.fc(x)
        x = self.sigmoid(x)

        return x.squeeze()
    

class ModifiedGRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, cnn_out_channels=32, cnn_kernel_size=3, pool_kernel_size=2, dropout_prob=0.2):
        super(ModifiedGRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # CNN layer
        self.cnn = nn.Conv1d(input_size, cnn_out_channels, cnn_kernel_size)
        self.maxpool = nn.MaxPool1d(pool_kernel_size)
        self.dropout_cnn = nn.Dropout(dropout_prob)

        self.gru = nn.GRU(cnn_out_channels, hidden_size, num_layers, batch_first=True)
        self.dropout_gru = nn.Dropout(dropout_prob)

        #self.fc = nn.Linear(hidden_size, 1)
        self.fc = nn.Linear(hidden_size, 2)  # Change this line
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x, lengths):
        x = x.permute(0, 2, 1)  # Reshape input for Conv1d

        # CNN layer
        x = self.dropout_cnn(self.relu(self.cnn(x)))
        x = self.maxpool(x)
        x = x.permute(0, 2, 1)

        # GRU layer
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        output, h_n = self.gru(x, h0)
        x = self.dropout_gru(h_n[-1])

        # Fully connected layer
        x = self.fc(x)
        x = self.sigmoid(x)

        return x.squeeze()
    
class late_fusion_hidden_layer(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes,GRU_eyes, head_gru, face_gru, dropout):
        super(late_fusion_hidden_layer, self).__init__()
        self.linear_layer = nn.Linear(input_size, num_classes)
        self.sigmoid = nn.Sigmoid()
        self.GRU_eyes = GRU_eyes
        self.head_gru = head_gru
        self.face = face_gru
       


        for param in self.GRU_eyes.parameters():
            param.requires_grad = False
        for param in self.head_gru.parameters():
            param.requires_grad = False
        for param in self.face.parameters():
            param.requires_grad = False

    
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 32)
        self.fc4 = nn.Linear(32, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x_lengths):
        x_head, x_eyes, x_face = x
        x_head_len, x_eyes_len, x_face_len = x_lengths
        h_head = self.head_gru(x_head, x_head_len)
        h_eyes = self.GRU_eyes(x_eyes, x_eyes_len)
        h_face = self.face(x_face, x_face_len)
        #print(h_eyes.size())
        #print(h_head.size())
        hidden_concat = torch.cat((h_head,h_eyes,h_face), dim = 1)
        out = self.dropout(self.relu(self.fc1(hidden_concat)))
        out = self.dropout(self.relu(self.fc2(out)))
        out = self.dropout(self.relu(self.fc3(out)))
        out = self.fc4(out)
        out = self.sigmoid(out)
        return out.squeeze()
    

class late_fusion_linear(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, GRU_eyes, head_gru):
        super(late_fusion_linear, self).__init__()

        self.sigmoid = nn.Sigmoid()
        self.GRU_eyes = GRU_eyes
        self.head_gru =head_gru

        for param in self.GRU_eyes.parameters():
            param.requires_grad = False
        for param in self.head_gru.parameters():
            param.requires_grad = False

        self.fc1 = nn.Linear(input_size, num_classes)


    def forward(self, x, x_lengths):
        x_head, x_eyes = x
        x_head_len, x_eyes_len = x_lengths
        probs_body = self.head_gru(x_head, x_head_len)
        probs_eyes = self.GRU_eyes(x_eyes, x_eyes_len)
        hidden_concat = torch.cat((probs_body,probs_eyes), dim = 1)
        out = self.fc1(hidden_concat)
        out = self.sigmoid(out)
        return out.squeeze()
    

class ModifiedGRUModel_hidden_output(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, cnn_out_channels=32, cnn_kernel_size=3, pool_kernel_size=2, dropout_prob=0.2):
        super(ModifiedGRUModel_hidden_output, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # CNN layer
        self.cnn = nn.Conv1d(input_size, cnn_out_channels, cnn_kernel_size)
        self.maxpool = nn.MaxPool1d(pool_kernel_size)
        self.dropout_cnn = nn.Dropout(dropout_prob)

        self.gru = nn.GRU(cnn_out_channels, hidden_size, num_layers, batch_first=True)
        self.dropout_gru = nn.Dropout(dropout_prob)

        #self.fc = nn.Linear(hidden_size, 1)
        self.fc = nn.Linear(hidden_size, 2)  # Change this line
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x, lengths):
        x = x.permute(0, 2, 1)  # Reshape input for Conv1d

        # CNN layer
        x = self.dropout_cnn(self.relu(self.cnn(x)))
        x = self.maxpool(x)
        x = x.permute(0, 2, 1)

        # GRU layer
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        output, h_n = self.gru(x, h0)
        x_last = self.dropout_gru(h_n[-1])

        # Fully connected layer
        x = self.fc(x_last)
        x = self.sigmoid(x)

        return x_last

class GRUModel_last_output(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(GRUModel_last_output, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, lengths):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Pack the padded sequences
        packed_input = rnn_utils.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        out, hn =  self.gru(packed_input, h0)

        # Unpack the packed sequence
        unpacked_out, _ = rnn_utils.pad_packed_sequence(out, batch_first=True)

        # Take the output of the last time step
        last_output = unpacked_out[torch.arange(len(unpacked_out)), lengths - 1]

        out = self.fc(last_output)
        out = self.sigmoid(out)

        return hn[-1]