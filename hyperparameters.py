from losses import *

################################################################################
# Hyperparameters
################################################################################

# Leaky ReLU
alpha = 0.1

# Input Image Dimensions
# (rows, cols, depth, channels)
input_dimensions = (128,128,1)
dimensions = (128,128)

# Dropout probability
dropout = 0.5

# Training parameters
num_initial_filters = 32
batchnorm = True

# batch_size must be a multiple of num_gpu to ensure GPUs are not starved of data
num_gpu = 1
batch_size = 64
steps_per_epoch = 8

learning_rate = 0.0001
loss = tversky_loss
metrics = [dice_coef]
epochs = 2500

# Paths
checkpoint_path = "checkpoint"
log_path = "log.txt"
save_path = "model"
