import torch


# Device Parameters
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# DEVICE = 'cpu'
# Data Loader parameters
BATCH_SIZE = 8
NUM_WORKERS = 0

# Generator hyper parameters
NUM_CHANNELS = 64
NUM_BLOCKS = 16
GEN_LR = 0.01

# Discriminator parameteres
DISC_LR = 0.01


# Checkpoint Locations

GEN_CHK = './models/gen_chk'
DISC_CHK = './models/disc_chk'


EPOCHS = 100
LEARNING_RATE = 1e-5
LOAD = True