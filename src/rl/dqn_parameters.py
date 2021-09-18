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

# buffer params
UNBALANCE_P = 0.8 
BUFFER_UNBALANCE_GAP = 0.5

# training parameters
STD_DEV = 0.2
BATCH_SIZE = 100
BUFFER_SIZE = 1e6
TOTAL_EPISODES = 10000
Q_LR = 1e-5
CRITIC_LR = 1e-3
COST_LR = 1e-3
ACTOR_LR = 1e-4
WARM_UP = 0

# Safety Parameters
NUM_STEPS = 100
COST_BOUND = 10