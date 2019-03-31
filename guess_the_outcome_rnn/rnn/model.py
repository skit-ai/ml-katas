import torch
import torch.nn as nn


class RNN(nn.Module):
    # https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.rel1 = nn.ReLU()
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.rel2 = nn.ReLU()
        self.softmax = nn.LogSoftmax(dim=1)
        self.learning_rate = 0.001

    def forward(self, input_, hidden):
        combined = torch.cat((input_, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)

