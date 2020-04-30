import pydicom
import matplotlib as plt
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from mask_functions import rle2mask

ID = "path/to/file"
img = pydicom.dcmread(ID).pixel_array
print(img.dtype)
img = np.reshape(img, (1, 1024, 1024, 1))
img = tf.convert_to_tensor(img, dtype=tf.uint8)
ksizes = [1, 64, 64, 1]
strides = [1, 64, 64, 1]  # how far the centres of two patches must be --> overlap area
rates = [1, 1, 1, 1]  # dilation rate of patches
img_patches = tf.image.extract_patches(img, ksizes, strides, rates, padding="VALID")
img_patches = np.reshape(img_patches, (256, 64, 64, 1))
plt.figure(figsize=(10,10))
plimg = np.reshape(img_patches[100,:,:,:], (64,64))
plt.imshow(plimg, cmap=plt.cm.bone)
plt.show()
