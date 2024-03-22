import torch #pytorch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset
 
class late_fusion_hidden_layer(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, GRU_eyes_hidden_output, body_lstm_hidden_output, face_lstm_hidden_output):
        super(late_fusion_hidden_layer, self).__init__()
        self.linear_layer = nn.Linear(input_size, num_classes)
        self.sigmoid = nn.Sigmoid()
        self.GRU_eyes = GRU_eyes_hidden_output
        self.body_lstm =body_lstm_hidden_output
        self.face_lstm = face_lstm_hidden_output
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x_body, x_face, x_eyes = x
        h_body = self.body_lstm(x_body)
        h_face = self.face_lstm(x_face)
        h_eyes = self.GRU_eyes(x_eyes)
        hidden_concat = '....'
        out = self.dropout(self.relu(self.fc1(hidden_concat)))
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out.squeeze()
    

class late_fusion_probs(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, GRU_eyes, body_lstm, face_lstm):
        super(late_fusion_probs, self).__init__()
        self.linear_layer = nn.Linear(input_size, num_classes)
        self.sigmoid = nn.Sigmoid()
        self.GRU_eyes = GRU_eyes
        self.body_lstm =body_lstm
        self.face_lstm = face_lstm
        self.fc1 = nn.Linear(input_size, num_classes)


    def forward(self, x):
        x_body, x_face, x_eyes = x
        probs_body = self.body_lstm(x_body)
        probs_face = self.face_lstm(x_face)
        probs_eyes = self.GRU_eyes(x_eyes)
        hidden_concat = '....'
        out = self.fc1(hidden_concat)
        out = self.sigmoid(out)
        return out.squeeze()


class lstm_body(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(lstm_body, self).__init__()
        self.num_classes = num_classes #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.seq_length = seq_length #sequence length

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers,
                                bidirectional=False, batch_first=True) #lstm
        self.fc_1 =  nn.Linear(hidden_size*self.num_layers, hidden_size) #fully connected 1
        self.fc = nn.Linear(hidden_size, num_classes) #fully connected last layer
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(hidden_size*self.num_layers, affine=True)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self,x):
        #h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) #hidden state
        #c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) #internal state
        # Propagate input through LSTM
        #print(x.shape)

        output, (hn, cn) = self.lstm(x) #lstm with input, hidden, and internal state
        #print(hn.shape)
        hn = hn.view(-1, self.num_layers*self.hidden_size) #reshaping the data for Dense layer next
        #print(hn.shape)
        out = self.bn1(hn)
        #out = self.relu(hn)
        out = self.dropout(self.relu(self.fc_1(out))) #first Dense
        #out = self.relu(out) #relu
        out = self.fc(out) #Final Output
        #out = self.sigmoid(out)
        #print(torch.squeeze(out).shape)
        return torch.squeeze(out)
    
class lstm_face(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(lstm_face, self).__init__()
        self.num_classes = num_classes #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.seq_length = seq_length #sequence length

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers,
                                bidirectional=False, batch_first=True) #lstm
        self.fc_1 =  nn.Linear(hidden_size*self.num_layers, hidden_size) #fully connected 1
        self.fc = nn.Linear(hidden_size, num_classes) #fully connected last layer
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(hidden_size*self.num_layers, affine=True)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self,x):
        #h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) #hidden state
        #c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) #internal state
        # Propagate input through LSTM
        #print(x.shape)

        output, (hn, cn) = self.lstm(x) #lstm with input, hidden, and internal state
        #print(hn.shape)
        hn = hn.view(-1, self.num_layers*self.hidden_size) #reshaping the data for Dense layer next
        #print(hn.shape)
        out = self.bn1(hn)
        #out = self.relu(hn)
        out = self.dropout(self.relu(self.fc_1(out))) #first Dense
        #out = self.relu(out) #relu
        out = self.fc(out) #Final Output
        #out = self.sigmoid(out)
        #print(torch.squeeze(out).shape)
        return torch.squeeze(out)

    
class GRU_eyes(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, cnn_out_channels=32, cnn_kernel_size=3, pool_kernel_size=2, dropout_prob=0.2):
        super(GRU_eyes, self).__init__()
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
