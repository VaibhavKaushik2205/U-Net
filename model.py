## Libraries
import tensorflow as tf
import tensorflow.keras as keras
from keras.utils import np_utils 
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import MaxPooling2D, Add
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import concatenate, Cropping2D
from tensorflow.keras.optimizers import Adam, SGD
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


## Create model
#Utility_Blocks
def contracting_block(net, out_channels, stage, padding, k_size=3):
    
    net = Conv2D(filters=out_channels, kernel_size=(k_size,k_size),
                 name = 'contracting_path_' + stage + '_a',
                 strides=(1,1), padding=padding)(net)
    net = Activation('relu')(net)
    net = BatchNormalization()(net)
    
    net = Conv2D(filters=out_channels, kernel_size=(k_size,k_size),
                 name = 'contracting_path_' + stage + '_b',
                 padding=padding)(net)
    net = Activation('relu')(net)
    net = BatchNormalization()(net)
    
    return net

def expanding_block(net, conv_layer, out_channels, img_size, conv_size, padding, stage, k_size=3):
    
    net = Conv2DTranspose(filters=out_channels, kernel_size=(2,2),
                          padding='same', strides=(2,2),
                          name = 'expanding_block_' + stage + '_tranpose')(net)
    
    ## Use Cropping2D for conv_layer in case padding is 'valid' for convolutions in contracting block
    if padding == 'valid':
        crop_size = int((conv_size-img_size)/2)
        conv_layer = Cropping2D(cropping=(crop_size))(conv_layer)
    
    net = concatenate([net, conv_layer], axis=-1)
    
    net = Conv2D(filters=out_channels, kernel_size=(k_size,k_size),
                 name = 'expanding_block_' + stage + '_a',
                 padding=padding)(net)
    net = Activation('relu')(net)
    net = BatchNormalization()(net)
    
    net = Conv2D(filters=out_channels, kernel_size=(k_size,k_size),
                 name = 'expanding_block_' + stage + '_b',
                 padding=padding)(net)
    net = Activation('relu')(net)
    net = BatchNormalization()(net)
    
    return net
    
#Contracting_Layer
def contract(inputs, img_size, padding, filters):
    
    # Block 1
    net1 = contracting_block(inputs, out_channels=filters[0], stage='1', padding=padding)
    net = MaxPooling2D((2,2), name = 'max_pooling_1')(net1)
    
    # Block 2
    net2 = contracting_block(net, out_channels=filters[1], stage='2', padding=padding)
    net = MaxPooling2D((2,2), name = 'max_pooling_2')(net2)
    
    # Block 3
    net3 = contracting_block(net, out_channels=filters[2], stage='3', padding=padding)
    net = MaxPooling2D((2,2), name = 'max_pooling_3')(net3)
    
    # Block 4
    net4 = contracting_block(net, out_channels=filters[3], stage='4', padding=padding)
    net = MaxPooling2D((2,2), name = 'max_pooling_4')(net4)
    
    if padding=='same':
        img_size /= 4
    else:
        for i in range(4):
            img_size -= 4
            img_size /= 2
        
    print('contract ' + str(img_size))
    
    return net, net1, net2, net3, net4, img_size

#Expanding_Layer
def expand(net, net1, net2, net3, net4, img_size, padding, filters):
    
    if padding=='same':
        conv_size = img_size
    else:
        conv_size = ((img_size/2) + 4) * 2
    
    # Up-Sample 1
    print('expand_' + str(img_size) + ' ' + str(conv_size))
    net = expanding_block(net, net4, filters[3], img_size, conv_size, padding, stage='1')
    
    if padding=='same':
        img_size *= 2
        conv_size = img_size
    else:
        img_size = (img_size-4) * 2
        conv_size = (conv_size+4) * 2
    
    # Up-Sample 2
    print('expand_' + str(img_size) + ' ' + str(conv_size))
    net = expanding_block(net, net3, filters[2], img_size, conv_size, padding, stage='2')
    
    if padding=='same':
        img_size *= 2
        conv_size = img_size
    else:
        img_size = (img_size-4) * 2
        conv_size = (conv_size+4) * 2
    
    # Up-Sample 3
    print('expand_' + str(img_size) + '_' + str(conv_size))
    net = expanding_block(net, net2, filters[1], img_size, conv_size, padding, stage='3')
    
    if padding=='same':
        img_size *= 2
        conv_size = img_size
    else:
        img_size = (img_size-4) * 2
        conv_size = (conv_size+4) * 2
    
    # Up-Sample 4
    print('expand_' + str(img_size) + '_' + str(conv_size))
    net = expanding_block(net, net1, filters[0], img_size, conv_size, padding, stage='4')
    
    return net, img_size

#UNet
def UNet(input_shape, classes, img_size, filters, padding):
    """
    Implementation of the UNet having the following architecture:
    CONTRACTING_LAYER: (CONV2D*2 -> MAXPOOL) *4
    BOTTLENECK_LAYER:  CONV2D -> CONV2D
    EXPANDING_LAYER: (UPSCALING(2x2CONV) -> CONV2D*2) *4

    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- output channels of model
    img_size -- integer, size of input images
    filters -- list of filters in each layer
    padding -- string, 'same' or 'valid'

    Returns:
    model -- a Model() instance in Keras
    """
    
    # Define a tensor with input shape
    inputs = Input(input_shape)
    
    # Select padding
    padding = padding
    
    # Contracting Layer
    net, net1, net2, net3, net4, img_size = contract(inputs, img_size, padding, filters)
    
    # BottleNeck Layer
    net = Conv2D(filters=filters[4], kernel_size=(3,3),
                name = 'bottleNeck_1', padding=padding, 
                strides=(1,1))(net)
    net = Activation('relu')(net)
    net = BatchNormalization()(net)
    
    net = Conv2D(filters=filters[4], kernel_size=(3,3),
                name = 'bottleNeck_2', padding=padding, 
                strides=(1,1))(net)
    net = Activation('relu')(net)
    net = BatchNormalization()(net)
    
    if padding == 'valid':
        img_size = (img_size-4) * 2
        
    print (img_size)
    
    # Expanding Layer
    net, img_size = expand(net, net1, net2, net3, net4, img_size, padding, filters)
    
    # Final Conv
    net = Conv2D(filters=classes, kernel_size=(1,1),
                 name = 'output_convolution')(net)
    net = Activation('sigmoid')(net)
    net = BatchNormalization()(net)
    
    model = Model(inputs=inputs, outputs=net, name='U-Net')
    
    return model