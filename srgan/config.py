import torch


# Device Parameters
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Data Loader parameters
BATCH_SIZE = 16
NUM_WORKERS = 0

# Generator hyper parameters
NUM_CHANNELS = 64
NUM_BLOCKS = 16
GEN_LR = 0.01

# Discriminator parameteres
DISC_LR = 0.01