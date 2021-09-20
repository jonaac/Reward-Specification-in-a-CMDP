"""
Common definitions of variables that can be used across files
"""

from tensorflow.keras.initializers import glorot_normal

# general parameters
CHECKPOINTS_PATH = "../../savedmodels/"
TF_LOG_DIR = './logs/DDPG/'

# CLRM parameters
GAMMA = 0.9  # for the temporal difference
RHO = 0.001  # to update the target networks
KERNEL_INITIALIZER = glorot_normal()

# training parameters
STD_DEV = 0.2
BATCH_SIZE = 32  # * n
BUFFER_SIZE = 50000  # * n
TOTAL_STEPS = 1e06
Q_LR = 1e-5
CRITIC_LR = 1e-5
COST_LR = 1e-5
WARM_UP = 0

# Safety Parameters
NUM_STEPS = 100
COST_BOUND = 10
