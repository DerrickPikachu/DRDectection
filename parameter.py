import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

epoch = 5
batch_size = 4
learning_rate = 0.1
loss_fn = torch.nn.CrossEntropyLoss()

# TODO: Need to survey the effect of these parameters
momentum = 0.9
weight_decay = 5e-4
