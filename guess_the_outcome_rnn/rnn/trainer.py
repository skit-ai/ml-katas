import random
import time
import math
import torch

from torch import nn
from guess_the_outcome_rnn.rnn.model import RNN
from guess_the_outcome_rnn.proc.generate_io import n_letters, line_to_tensor, category_from_output


# This is kinda unfair, knowing this makes the blackbox white :(
expected_outcomes = 2

# Attempt 1
n_hidden = 100

# Create an rnn model
rnn = RNN(n_letters, n_hidden, expected_outcomes)

# Hyper-parameters
learning_rate = 0.05

# Loss function
criterion = nn.NLLLoss()


def __time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def random_training_example(input_value, output_value):
    category = random.randint(0, 1)
    category_tensor = torch.tensor([expected_outcomes.index(output_value)], dtype=torch.long)
    line_tensor = line_to_tensor(input_value)
    return category, category_tensor, line_tensor


def train_step(category_tensor, line_tensor):
    hidden = rnn.init_hidden()

    rnn.zero_grad()
    output = None
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)
    loss.backward()

    torch.nn.utils.clip_grad_norm_(rnn.parameters(), 1)

    # Add parameters' gradients to their values, multiplied by learning rate
    for p in rnn.parameters():
        p.data.add_(-learning_rate, p.grad.data)

    return output, loss.item()


def train(epochs, i, input_value, output_value):
    print_every = 100
    start = time.time()
    category, category_tensor, line_tensor = random_training_example(input_value, output_value)
    output, loss = train_step(category_tensor, line_tensor)

    # Print iter number, loss, name and guess
    if i % print_every == 0:
        print(line_tensor.shape)
        guess, guess_i = category_from_output(output)
        correct = '✓' if guess == category else '✗ (%s)' % category
        print('%d %d%% (%s) loss=%.4f input=(%s) output=(%s) result=%s' %
              (i, i / epochs * 100, __time_since(start), loss, input_value, guess, correct)
              )

    return loss
