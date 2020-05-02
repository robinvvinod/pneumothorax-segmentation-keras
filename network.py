# yapf: disable
from keras.models import Model
from keras.layers.convolutional import Conv2D
from layers2D import *

# Use the functions provided in layers3D to build the network

def network(input_img, n_filters=16, dropout=0.5, batchnorm=True):

    # contracting path
    
    c0 = residual_block(input_img, n_filters=n_filters, batchnorm=batchnorm, strides=1, recurrent=2)

    c1 = residual_block(c0, n_filters=n_filters * 2, batchnorm=batchnorm, strides=2, recurrent=2)

    c2 = residual_block(c1, n_filters=n_filters * 4, batchnorm=batchnorm, strides=2, recurrent=2)

    c3 = residual_block(c2, n_filters=n_filters * 8, batchnorm=batchnorm, strides=1, recurrent=2)
    
    # bridge
    
    b0 = residual_block(c3, n_filters=n_filters * 16, batchnorm=batchnorm, strides=2, recurrent=2)

    # expansive path
    
    attn0 = AttnGatingBlock(c3, b0, n_filters * 16)
    u0 = transpose_block(b0, attn0, n_filters=n_filters * 8)
    
    attn1 = AttnGatingBlock(c2, u0, n_filters * 8)
    u1 = transpose_block(u0, attn1, n_filters=n_filters * 4)
    
    attn2 = AttnGatingBlock(c1, u1, n_filters * 4)
    u2 = transpose_block(u1, attn2, n_filters=n_filters * 2)
    
    u3 = transpose_block(u2, c0, n_filters=n_filters)  

    outputs = Conv2D(filters=1, kernel_size=1, strides=1, activation='sigmoid')(d3)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model
