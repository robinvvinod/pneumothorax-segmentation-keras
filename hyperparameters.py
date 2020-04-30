from losses import *

################################################################################
# Hyperparameters
################################################################################

# Leaky ReLU
alpha = 0.1

# Input Image Dimensions
# (rows, cols, depth, channels)
input_dimensions = (64,64,1)
dimensions = (64,64)

# Dropout probability
dropout = 0.5

# Training parameters
num_initial_filters = 32
batchnorm = True

# batch_size must be a multiple of num_gpu to ensure GPUs are not starved of data
num_gpu = 1
batch_size = 1
steps_per_epoch = 1

learning_rate = 0.0001
loss = tversky_loss
metrics = [dice_coef]
epochs = 20

# Paths
checkpoint_path = "/Users/RobinVinod/Documents/Coding/ML/pneumothorax-segmentation/data/checkpoint"
log_path = "/Users/RobinVinod/Documents/Coding/ML/pneumothorax-segmentation/data/log.txt"
save_path = "/Users/RobinVinod/Documents/Coding/ML/pneumothorax-segmentation/data/model"
train_path = "/Users/RobinVinod/Documents/Coding/ML/pneumothorax-segmentation/data/stage_2_images"
test_path = ""
