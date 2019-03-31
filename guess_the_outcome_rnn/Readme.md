# RNN: Guess the outcome of a blackbox

## Problem
Consider a consistent program as a black-box, assume: 
- For same inputs the result will always be the same (there is no impact of state or time)
- The program may throw exceptions which can be considered as an output.
- Many inputs may lead to same output.

Using this system as a dataset, try to train a Recurrent Neural Network which is able to 
guess the output.


## Challenges 
1. Training data is not present at the start of the exercise, this is an online training problem.
2. Learning the challenges with training RNNs, discover the meaning of unstable in terms of training.
3. Parameter tuning, controlling growth of weights as the training progresses.
4. Creating a healthy batch of examples in a system where training data is available as a stream.


## Tools
- Pytorch 1.0.1
- Python 3.5.6

## General steps
Every attempt at this problem must have:

1. Strategy outline.
2. Training step with notes on hyper-parameters, loss-plot and a saved model.
3. Validation step with a comparison of train-test accuracy.


## Observations
1. Cite reasons for choosing a strategy 'random' is acceptable for first few attempts.
2. Take notes on loss per epoch during training.
3. Infer the message behind a loss plot.
4. Compare train-test accuracies and note down observations/guesses for the behaviour.
