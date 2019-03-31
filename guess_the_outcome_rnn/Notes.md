Attempt 1
===

## Strategy outline.
- Creating a 1-hot encoded vector for each character in a string.
- Learning rate = `0.05` 
- Number of hidden layers = `1` 
- Number of neurons in a hidden layer = `100`
- Loss function = Negative log loss
- Directly feed the inputs to the blockbox via model trainer

### Expectations
- This model may not fare well since there are a lot of parameters which are randomly chosen
- Directly feeding the inputs to the model seems like a bad idea, the model may not receive
a good split of the classes to be predicted.
