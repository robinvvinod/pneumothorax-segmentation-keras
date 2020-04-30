# yapf: disable
from keras.models import Model
from keras.layers.convolutional import Conv2D
from keras.layers import SpatialDropout2D
from layers2D import *

# Use the functions provided in layers3D to build the network

def network(input_img, n_filters=16, dropout=0.5, batchnorm=True):

    # contracting path
    
    c0 = residual_block(input_img, n_filters=n_filters, batchnorm=batchnorm, strides=1, recurrent=2)
    p0 = SpatialDropout2D(dropout * 0.5)(c0)

    c1 = residual_block(p0, n_filters=n_filters * 2, batchnorm=batchnorm, strides=2, recurrent=2)
    p1 = SpatialDropout2D(dropout)(c1)

    c2 = residual_block(p1, n_filters=n_filters * 4, batchnorm=batchnorm, strides=2, recurrent=2)
    p2 = SpatialDropout2D(dropout)(c2)

    c3 = residual_block(p2, n_filters=n_filters * 8, batchnorm=batchnorm, strides=1, recurrent=2)
    p3 = SpatialDropout2D(dropout)(c3)
    
    # bridge
    
    b0 = residual_block(p3, n_filters=n_filters * 16, batchnorm=batchnorm, strides=2, recurrent=2)

    # expansive path
    
    gating = UnetGatingSignal(b0, batchnorm=batchnorm)
    attn0 = AttnGatingBlock(p3, gating, n_filters * 16)
    u0 = transpose_block(b0, attn0, n_filters=n_filters * 8)
    d0 = SpatialDropout2D(dropout)(u0)
    
    gating = UnetGatingSignal(d0, batchnorm=batchnorm)
    attn1 = AttnGatingBlock(p2, gating, n_filters * 8)
    u1 = transpose_block(d0, attn1, n_filters=n_filters * 4)
    d1 = SpatialDropout2D(dropout)(u1)
    
    gating = UnetGatingSignal(d1, batchnorm=batchnorm)
    attn2 = AttnGatingBlock(p1, gating, n_filters * 4)
    u2 = transpose_block(d1, attn2, n_filters=n_filters * 2)
    d2 = SpatialDropout2D(dropout)(u2)
    
    u3 = transpose_block(d2, p0, n_filters=n_filters)  
    d3 = SpatialDropout2D(dropout)(u3)

    outputs = Conv2D(filters=1, kernel_size=1, strides=1, activation='sigmoid')(d3)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model
