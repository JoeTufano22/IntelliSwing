import torch
import torch.nn as nn
from torch.autograd import Variable
from MobileNetV2 import MobileNetV2


class EventDetector(nn.Module):
    def __init__(self, pretrain, width_mult, lstm_layers, lstm_hidden, bidirectional=True, dropout=True):
        super(EventDetector, self).__init__()
        self.width_mult = width_mult
        self.lstm_layers = lstm_layers
        self.lstm_hidden = lstm_hidden
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.device = None

        net = MobileNetV2(width_mult=width_mult)
        if pretrain:
            try:
                state_dict_mobilenet = torch.load('mobilenet_v2.pth.tar', weights_only=False)
                net.load_state_dict(state_dict_mobilenet, strict=False)
                print("Loaded MobileNetV2 pretrained weights (partial match)")
            except Exception as e:
                print(f"Warning: Could not load pretrained weights: {e}")
                print("Training from scratch...")

        self.cnn = nn.Sequential(*list(net.children())[0][:19])
        self.rnn = nn.LSTM(int(1280*width_mult if width_mult > 1.0 else 1280),
                           self.lstm_hidden, self.lstm_layers,
                           batch_first=True, bidirectional=bidirectional)
        if self.bidirectional:
            self.lin = nn.Linear(2*self.lstm_hidden, 9)
        else:
            self.lin = nn.Linear(self.lstm_hidden, 9)
        if self.dropout:
            self.drop = nn.Dropout(0.5)

    def init_hidden(self, batch_size):
        device = next(self.parameters()).device
        if self.bidirectional:
            return (Variable(torch.zeros(2*self.lstm_layers, batch_size, self.lstm_hidden, device=device), requires_grad=True),
                    Variable(torch.zeros(2*self.lstm_layers, batch_size, self.lstm_hidden, device=device), requires_grad=True))
        else:
            return (Variable(torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden, device=device), requires_grad=True),
                    Variable(torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden, device=device), requires_grad=True))

    def forward(self, x, lengths=None):
        batch_size, timesteps, C, H, W = x.size()
        self.hidden = self.init_hidden(batch_size)

        c_in = x.view(batch_size * timesteps, C, H, W)
        c_out = self.cnn(c_in)
        c_out = c_out.mean(3).mean(2)
        if self.dropout:
            c_out = self.drop(c_out)

        r_in = c_out.view(batch_size, timesteps, -1)
        r_out, states = self.rnn(r_in, self.hidden)
        out = self.lin(r_out)
        out = out.view(batch_size*timesteps,9)

        return out



