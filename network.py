# yapf: disable
from keras.models import Model
from keras.layers.convolutional import Conv2D
from keras.layers import SpatialDropout2D
from layers2D import *

# Use the functions provided in layers3D to build the network

def network(input_img, n_filters=16, dropout=0.5, batchnorm=True):

    # contracting path
    
    c0 = inception_block(input_img, n_filters=n_filters, batchnorm=batchnorm, strides=1, recurrent=2, layers=((3,2),(5,1)))  # 64x64
    p0 = SpatialDropout2D(dropout * 0.5)(c0)

    c1 = inception_block(p0, n_filters=n_filters * 2, batchnorm=batchnorm, strides=2, recurrent=2, layers=((3,2),(5,1)))  # 32x32
    p1 = SpatialDropout2D(dropout)(c1)

    c2 = inception_block(p1, n_filters=n_filters * 4, batchnorm=batchnorm, strides=2, recurrent=2, layers=((3,2),(5,1)))  # 16x16
    p2 = SpatialDropout2D(dropout)(c2)

    c3 = inception_block(p2, n_filters=n_filters * 8, batchnorm=batchnorm, strides=1, recurrent=2, layers=((3,2),(5,1)))  # 16x16
    p3 = SpatialDropout2D(dropout)(c3)
    
    # bridge
    
    b0 = inception_block(p3, n_filters=n_filters * 16, batchnorm=batchnorm, strides=2, recurrent=2, layers=((3,2),(5,1)))  # 8x8

    # expansive path
    
    gating = UnetGatingSignal(b0, batchnorm=batchnorm)
    attn0 = AttnGatingBlock(p3, gating, n_filters * 16)
    u0 = transpose_block(b0, attn0, n_filters=n_filters * 8)  # 16x16
    d0 = SpatialDropout2D(dropout)(u0)
    
    gating = UnetGatingSignal(d0, batchnorm=batchnorm)
    attn1 = AttnGatingBlock(p2, gating, n_filters * 8)
    u1 = transpose_block(d0, attn1, n_filters=n_filters * 4)  # 16x16
    d1 = SpatialDropout2D(dropout)(u1)
    
    gating = UnetGatingSignal(d1, batchnorm=batchnorm)
    attn2 = AttnGatingBlock(p1, gating, n_filters * 4)
    u2 = transpose_block(d1, attn2, n_filters=n_filters * 2)  # 32x32
    d2 = SpatialDropout2D(dropout)(u2)
    
    u3 = transpose_block(d2, p0, n_filters=n_filters)  # 64x64
    d3 = SpatialDropout2D(dropout)(u3)

    outputs = Conv2D(filters=1, kernel_size=1, strides=1, activation='sigmoid')(d3)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model
