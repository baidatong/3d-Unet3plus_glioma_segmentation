###
# Loss functions are modified from NiftyNet
###

import tensorflow as tf
# from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope
# from tensorpack.tfutils.summary import add_moving_summary
# from tensorpack.tfutils.argscope import argscope
# from tensorpack.tfutils.scope_utils import under_name_scope
#
from tensorpack.models import (
     BatchNorm, layer_register
 )
from custom_ops import BatchNorm3d, InstanceNorm5d
import numpy as np
import config
import tensorflow.contrib.slim as slim


PADDING = "SAME"
DATA_FORMAT = "channels_first"
s_num = 2
BASE_FILTER = 16
BASE_FILTER1 = 20
BASE_FILTER2 = 40
BASE_FILTER3 = 80/s_num
BASE_FILTER4 = 160/s_num
BASE_FILTER5 = 320/s_num
BASE_FILTER6 = 32
BASE_FILTER7 = 16
BASE_FILTER8 = 8
@layer_register(log_shape=True)
def unet3d(inputs):
    # filters; bacthnormaliztation dowsnsample upsample droupout ; full-scale surpervision; clssification;

    depth = config.DEPTH
    filters = []
    down_list = []
    deep_supervision = None
    layer0 =tf.layers.conv3d(inputs=inputs,
                             filters=BASE_FILTER1,   #20
                             kernel_size=(3, 3, 3),
                             strides=1,
                             padding=PADDING,
                             activation=lambda x, name=None: BN_Relu(x),
                             data_format=DATA_FORMAT,
                             name="layer1_1")

    # layer1 = tf.layers.conv3d(inputs = layer0, filters=BASE_FILTER1,   #20
    #                           kernel_size = (3,3,3),
    #                           strides=1,
    #                           padding=PADDING,
    #                           activation=lambda x, name=None: BN_Relu(x),
    #                           data_format=DATA_FORMAT,
    #                           name="layer1_1")
    layer1 = tf.layers.conv3d(inputs = layer0, filters=BASE_FILTER1,  #20
                              kernel_size = (3,3,3),
                              strides=1,
                              padding=PADDING,
                              activation=lambda x, name=None: BN_Relu(x),
                              data_format=DATA_FORMAT,
                              name="layer1_2")

    layer1 = tf.add(layer0,layer1)  #  z x y basefilter1
    # down sampling1
    down1 = tf.layers.conv3d(inputs = layer1,
                                     filters = BASE_FILTER2,   ##  z/2 x/2 y/2 basefilter2
                                     kernel_size = (3, 3, 3),
                                     strides = (2, 2, 2),
                                     padding = PADDING,
                                     activation=lambda x, name=None: BN_Relu(x),
                                     data_format=DATA_FORMAT,
                                     name='down1')
    layer2 = tf.layers.conv3d(inputs = down1,
                                     filters = BASE_FILTER2,    # z/2 x/2 y/2 basefilter2
                                     kernel_size = (3, 3, 3),
                                     strides = 1,
                                     padding=PADDING,
                                     activation=lambda x, name=None: BN_Relu(x),
                                     data_format=DATA_FORMAT,
                                     name='layer2_1')
    layer2 = tf.layers.conv3d(inputs = layer2,
                                     filters = BASE_FILTER2,
                                     kernel_size = (3, 3, 3),
                                     strides = 1,
                                     padding=PADDING,
                                     activation=lambda x, name=None: BN_Relu(x),
                                     data_format=DATA_FORMAT,
                                     name='layer2_2')
    layer2 = tf.add(down1,layer2)     # z/2 x/2 y/2 basefilter2

    down2 = tf.layers.conv3d(inputs = layer2,                            # layer2:z/2 x/2 y/2 basefilter2
                                     filters = BASE_FILTER3,               # z/4 x/4 y/4 basefilter3
                                     kernel_size = (3, 3, 3),
                                     strides = (2, 2, 2),
                                     padding=PADDING,
                                     activation=lambda x, name=None: BN_Relu(x),
                                     data_format=DATA_FORMAT,
                                     name='down2')
    layer3 = tf.layers.conv3d(inputs = down2,                    # z/4 x/4 y/4 basefilter3
                                     filters = BASE_FILTER3,
                                     kernel_size = (3, 3, 3),
                                     strides = 1,
                                     padding=PADDING,
                                     activation=lambda x, name=None: BN_Relu(x),
                                     data_format=DATA_FORMAT,
                                     name='layer3_1')
    layer3 = tf.layers.conv3d(inputs = layer3,                     # z/4 x/4 y/4 basefilter3
                                     filters = BASE_FILTER3,
                                     kernel_size = (3, 3, 3),
                                     strides = 1,
                                     padding=PADDING,
                                     activation=lambda x, name=None: BN_Relu(x),
                                     data_format=DATA_FORMAT,
                                     name='layer3_2')
    layer3 = tf.layers.conv3d(inputs = layer3,                     # z/4 x/4 y/4 basefilter3
                                     filters = BASE_FILTER3,
                                     kernel_size = (3, 3, 3),
                                     strides = 1,
                                     padding=PADDING,
                                     activation=lambda x, name=None: BN_Relu(x),
                                     data_format=DATA_FORMAT,
                                     name='layer3_3')
    layer3 = tf.add(layer3,down2)                             # z/4 x/4 y/4 basefilter3

    down3 = tf.layers.conv3d(inputs = layer3,                 # z/8 x/8 y/8 basefilter4
                                     filters = BASE_FILTER4,
                                     kernel_size = (3, 3, 3),
                                     strides = (2, 2, 2),
                                     padding=PADDING,
                                     activation=lambda x, name=None: BN_Relu(x),
                                     data_format=DATA_FORMAT,
                                     name='down3')
    layer4 = tf.layers.conv3d(inputs = down3,                 # z/8 x/8 y/8 basefilter4
                                     filters = BASE_FILTER4,
                                     kernel_size = (3, 3, 3),
                                     strides = 1,
                                     padding=PADDING,
                                     activation=lambda x, name=None: BN_Relu(x),
                                     data_format=DATA_FORMAT,
                                     name='layer4_1')
    layer4 = tf.layers.conv3d(inputs = layer4,               # z/8 x/8 y/8 basefilter4
                                     filters = BASE_FILTER4,
                                     kernel_size = (3, 3, 3),
                                     strides = 1,
                                     padding=PADDING,
                                     activation=lambda x, name=None: BN_Relu(x),
                                     data_format=DATA_FORMAT,
                                     name='layer4_2')
    layer4 = tf.layers.conv3d(inputs = layer4,             # z/8 x/8 y/8 basefilter4
                              filters=BASE_FILTER4,
                              kernel_size=(3, 3, 3),
                              strides=1,
                              padding=PADDING,
                              activation=lambda x, name=None: BN_Relu(x),
                              data_format=DATA_FORMAT,
                              name='layer4_3')
    layer4 = tf.add(layer4,down3)                         # z/8 x/8 y/8 basefilter4
    down4 = tf.layers.conv3d(inputs = layer4,             # z/16 x/16 y/16 basefilter5
                                     filters = BASE_FILTER5,
                                     kernel_size = (3, 3, 3),
                                     strides = (2, 2, 2),
                                     padding=PADDING,
                                     activation=lambda x, name=None: BN_Relu(x),
                                     data_format=DATA_FORMAT,
                                     name='down4')

    layer5 = tf.layers.conv3d(inputs = down4 ,
                                     filters = BASE_FILTER5,  # z/16 x/16 y/16 basefilter5
                                     kernel_size = (3, 3, 3),
                                     strides = 1,
                                     padding=PADDING,
                                     activation=lambda x, name=None: BN_Relu(x),
                                     data_format=DATA_FORMAT,
                                     name='layer5_1')
    layer5 = tf.layers.conv3d(inputs = layer5,
                                     filters = BASE_FILTER5, # # z/16 x/16 y/16 basefilter5
                                     kernel_size = (3, 3, 3),
                                     strides = 1,
                                     padding=PADDING,
                                     activation=lambda x, name=None: BN_Relu(x),
                                     data_format=DATA_FORMAT,
                                     name='layer5_2')
    layer5 = tf.layers.conv3d(inputs = layer5,
                              filters = BASE_FILTER5,   # # z/16 x/16 y/16 basefilter5
                              kernel_size=(3, 3, 3),
                              strides=1,
                              padding=PADDING,
                              activation=lambda x, name=None: BN_Relu(x),
                              data_format=DATA_FORMAT,
                              name='layer5_3')
    layer5 = tf.add(layer5, down4)         # z/16 x/16 y/16 basefilter5
    upsample1_1 =  UnetUpsample(prefix='upsample1_1', l=layer5, num_filters = BASE_FILTER5, scale=2)  #num_filters,scale
    decoder1_1 = tf.layers.conv3d(inputs = upsample1_1,
                              filters = BASE_FILTER6,   #32
                              kernel_size=(3, 3, 3),
                              strides=1,
                              padding=PADDING,
                              activation=lambda x, name=None: BN_Relu(x),
                              data_format=DATA_FORMAT,
                              name='decoder1_1')
    decoder1_2 = tf.layers.conv3d(inputs = layer4,
                              filters = BASE_FILTER6,   #32
                              kernel_size=(3, 3, 3),
                              strides=1,
                              padding=PADDING,
                              activation=lambda x, name=None: BN_Relu(x),
                              data_format=DATA_FORMAT,
                              name='decoder1_2')
    decoder1_3 = max_pool3d(inputs=layer3,stride=2,depth=True)
    decoder1_3 = tf.layers.conv3d(inputs =decoder1_3,
                                 filters=BASE_FILTER6,  #
                                 kernel_size=(3, 3, 3),
                                 strides=1,
                                 padding=PADDING,
                                 activation=lambda x, name=None: BN_Relu(x),
                                 data_format=DATA_FORMAT,
                                 name='decoder1_3')
    decoder1_4 = max_pool3d(inputs = layer2,stride=4,depth=True)   #  layer2: z/2 x/2 y/2 basefilter2
                                                                      #  z/4 x/4 y/4
    #decode1_4 = tf.layers.max_pooling3d(inputs = decoder1_4,depth=True) #  z/8 x/8 y/8
    decoder1_4 = tf.layers.conv3d(inputs =decoder1_4,
                                 filters=BASE_FILTER6,  # #  z/8 x/8 y/8
                                 kernel_size=(3, 3, 3),
                                 strides=1,
                                 padding=PADDING,
                                 activation=lambda x, name=None: BN_Relu(x),
                                 data_format=DATA_FORMAT,
                                 name='decoder1_4')
    # decode1_5 = tf.layers.max_pooling3d(inputs=layer1,depth=True)   #z/2 x/2 y/2 basefilter1
    # decode1_5 = tf.layers.max_pooling3d(inputs=decode1_5,depth=True)  #z/4 x/4 y/4 basefilter1
    # decode1_5 = tf.layers.max_pooling3d(inputs=decode1_5,depth=True) #z/8 x/8 y/8 basefilter1
    decoder1_5 = max_pool3d(inputs=layer1,depth=True,stride=8)
    decoder1_5 = tf.layers.conv3d(inputs=decoder1_5,
                                 filters=BASE_FILTER6,               # 32
                                 kernel_size=(3, 3, 3),
                                 strides=1,
                                 padding=PADDING,
                                 activation=lambda x, name=None: BN_Relu(x),
                                 data_format=DATA_FORMAT,
                                 name='decoder1_5')

    decoder1 = tf.concat([decoder1_1,decoder1_2,decoder1_3,decoder1_4,decoder1_5],axis=1)   # 160
    decoder1 = tf.layers.conv3d(inputs=decoder1,
                                 filters=BASE_FILTER4,  #
                                 kernel_size=(3, 3, 3),
                                 strides=1,
                                 padding=PADDING,
                                 activation=lambda x, name=None: BN_Relu(x),
                                 data_format=DATA_FORMAT,
                                 name='decoder1_6')

    # UnetUpsample(prefix='upsample1_1', l=layer5, num_filters=BASE_FILTER5, scale=2)
    upsample2_1 = UnetUpsample(prefix='upsample2_1', l=layer5,  num_filters=BASE_FILTER7, scale=4)
    #upsample2_1 = UnetUpsample(prefix='upsample2_1_2', l=upsample2_1,BASE_FILTER4)
    decoder2_1 = tf.layers.conv3d(inputs=upsample2_1,
                                 filters=BASE_FILTER7,  #16
                                 kernel_size=(3, 3, 3),
                                 strides=1,
                                 padding=PADDING,
                                 activation=lambda x, name=None: BN_Relu(x),
                                 data_format=DATA_FORMAT,
                                 name='decoder2_1')
    decoder2_2 = UnetUpsample(prefix='upsample2_2', l = decoder1, num_filters=BASE_FILTER4, scale=2)
    decoder2_2 = tf.layers.conv3d(inputs = decoder2_2,
                                 filters=BASE_FILTER7,  #16
                                 kernel_size=(3, 3, 3),
                                 strides=1,
                                 padding=PADDING,
                                 activation=lambda x, name=None: BN_Relu(x),
                                 data_format=DATA_FORMAT,
                                 name='decoder2_2')
    decoder2_3 = tf.layers.conv3d(inputs = layer3,
                                 filters=BASE_FILTER7,  #16
                                 kernel_size=(3, 3, 3),
                                 strides=1,
                                 padding=PADDING,
                                 activation=lambda x, name=None: BN_Relu(x),
                                 data_format=DATA_FORMAT,
                                 name='decoder2_3')
    decoder2_4 = max_pool3d(inputs=layer2,depth=True,stride=2);     # tf.layers.max_pooling3d(inputs=layer2,depth=True)
    decoder2_4 = tf.layers.conv3d(inputs = decoder2_4,
                                 filters=BASE_FILTER7,  #16
                                 kernel_size=(3, 3, 3),
                                 strides=1,
                                 padding=PADDING,
                                 activation=lambda x, name=None: BN_Relu(x),
                                 data_format=DATA_FORMAT,
                                 name='decode2_4')
    # decoder2_5 = tf.layers.max_pooling3d(inputs=layer1,depth=True)   #
    # decoder2_5 = tf.layers.max_pooling3d(inputs=decoder2_5,depth=True)
    #decoder2_5 = tf.layers.max_pooling3d(inputs=layer1, depth=True)
    decoder2_5 = max_pool3d(inputs=layer1,depth=True,stride=4)
    decoder2_5 = tf.layers.conv3d(inputs = decoder2_5,
                                 filters=BASE_FILTER7,  #16
                                 kernel_size=(3, 3, 3),
                                 strides=1,
                                 padding=PADDING,
                                 activation=lambda x, name=None: BN_Relu(x),
                                 data_format=DATA_FORMAT,
                                 name='decoder2_5')
    decoder2 = tf.concat([decoder2_1,decoder2_2,decoder2_3,decoder2_4,decoder2_5],axis=1)
    decoder2 = tf.layers.conv3d(inputs = decoder2,
                                 filters=BASE_FILTER4,  #80
                                 kernel_size=(3, 3, 3),
                                 strides=1,
                                 padding=PADDING,
                                 activation=lambda x, name=None: BN_Relu(x),
                                 data_format=DATA_FORMAT,
                                 name='decoder2_6')
    upsample3_1 = UnetUpsample(prefix='upsample3_1', l = layer5, num_filters=BASE_FILTER4, scale=8)
    #
    # upsample3_1 = UnetUpsample(prefix='upsample3_1_2', l = upsample3_1, BASE_FILTER4)
    # upsample3_1 = UnetUpsample(prefix='upsample3_1', l = upsample3_1, BASE_FILTER4)

    decoder3_1 = tf.layers.conv3d(inputs = upsample3_1,
                                 filters=BASE_FILTER4,  #80
                                 kernel_size=(3, 3, 3),
                                 strides=1,
                                 padding=PADDING,
                                 activation=lambda x, name=None: BN_Relu(x),
                                 data_format=DATA_FORMAT,
                                 name='decoder3_1')
    # decoder3_2 = UnetUpsample(prefix='upsample3_2_1', l = decoder1, BASE_FILTER4)
    # decoder3_2 = UnetUpsample(prefix='upsample3_2', l = decoder3_2, BASE_FILTER4)
    decoder3_2 = UnetUpsample(prefix='upsample3_2', l=decode1, num_filters=BASE_FILTER4,scale=4)
    decoder3_2 = tf.layers.conv3d(inputs = decoder3_2,
                                 filters=BASE_FILTER4,  #80
                                 kernel_size=(3, 3, 3),
                                 strides=1,
                                 padding=PADDING,
                                 activation=lambda x, name=None: BN_Relu(x),
                                 data_format=DATA_FORMAT,
                                 name='decoder3_2')
    #decoder3_3 = UnetUpsample(prefix='upsample3_3', l = decode2, BASE_FILTER4)
    decoder3_3 = UnetUpsample(prefix='upsample3_3', l=decoder2, num_filters=BASE_FILTER4,scale=2)
    decoder3_3 = tf.layers.conv3d(inputs = decoder3_3,
                                 filters=BASE_FILTER4,  #80
                                 kernel_size=(3, 3, 3),
                                 strides=1,
                                 padding=PADDING,
                                 activation=lambda x, name=None: BN_Relu(x),
                                 data_format=DATA_FORMAT,
                                 name='decoder3_3')
    decoder3_4 = tf.layers.conv3d(inputs = layer2,
                                 filters=BASE_FILTER4,  #80
                                 kernel_size=(3, 3, 3),
                                 strides=1,
                                 padding=PADDING,
                                 activation=lambda x, name=None: BN_Relu(x),
                                 data_format=DATA_FORMAT,
                                 name='decoder3_4')
    decoder3_5 = max_pool3d(inputs=layer1,stride=2,depth=True)
    decoder3_5 = tf.layers.conv3d(inputs = decoder3_5,
                                 filters=BASE_FILTER4,  #16
                                 kernel_size=(3, 3, 3),
                                 strides=1,
                                 padding=PADDING,
                                 activation=lambda x, name=None: BN_Relu(x),
                                 data_format=DATA_FORMAT,
                                 name='decoder3_5')


    decoder3 = tf.concat([decoder3_1,decoder3_2,decoder3_3,decoder3_4,decoder3_5],axis=1)

    decoder3 = tf.layers.conv3d(inputs = decoder3,
                                 filters=BASE_FILTER4,  #80
                                 kernel_size=(3, 3, 3),
                                 strides=1,
                                 padding=PADDING,
                                 activation=lambda x, name=None: BN_Relu(x),
                                 data_format=DATA_FORMAT,
                                 name='decoder3_6')
    # upsample4_1 = UnetUpsample(prefix='upsample4_1_1', l = layer5, BASE_FILTER4)
    # upsample4_1 = UnetUpsample(prefix='upsample4_1_2', l = upsample4_1, BASE_FILTER4)
    # upsample4_1 = UnetUpsample(prefix='upsample4_1_3', l = upsample4_1, BASE_FILTER4)
    # upsample4_1 = UnetUpsample(prefix='upsample4_1', l = upsample4_1, BASE_FILTER4)
    upsample4_1 = UnetUpsample(prefix='upsample4_1', l = layer5, num_filters=BASE_FILTER4,scale=16)

    decoder4_1 = tf.layers.conv3d(inputs = upsample4_1,
                                 filters=BASE_FILTER4,  #80
                                 kernel_size=(3, 3, 3),
                                 strides=1,
                                 padding=PADDING,
                                 activation=lambda x, name=None: BN_Relu(x),
                                 data_format=DATA_FORMAT,
                                 name='decoder4_1')
    # upsample4_2 = UnetUpsample(prefix='upsample4_2_1', l = decode1, BASE_FILTER4)
    # upsample4_2 = UnetUpsample(prefix='upsample4_2_1', l = upsample4_2, BASE_FILTER4)
    # upsample4_2 = UnetUpsample(prefix='upsample4_2_2', l = upsample4_2, BASE_FILTER4)
    # decoder4_2  = UnetUpsample(prefix='decoder4_2', l = upsample4_2, BASE_FILTER4)
    decoder4_2  = UnetUpsample(prefix='decoder4_2', l = decoder1, num_filters=BASE_FILTER4,scale=8)

    decoder4_2 = tf.layers.conv3d(inputs = decoder4_2,
                                 filters=BASE_FILTER4,  #80
                                 kernel_size=(3, 3, 3),
                                 strides=1,
                                 padding=PADDING,
                                 activation=lambda x, name=None: BN_Relu(x),
                                 data_format=DATA_FORMAT,
                                 name='decoder4_2')
    # upsample4_3 = UnetUpsample(prefix='upsample4_3_1', l = decoder2, BASE_FILTER4)
    # upsample4_3 = UnetUpsample(prefix='upsample4_3', l = upsample4_3, BASE_FILTER4)
    upsample4_3 = UnetUpsample(prefix='upsample4_3', l=decoder2, num_filters=BASE_FILTER4,scale=4)
    decoder4_3 = tf.layers.conv3d(inputs = upsample4_3,
                                 filters=BASE_FILTER4,  #80
                                 kernel_size=(3, 3, 3),
                                 strides=1,
                                 padding=PADDING,
                                 activation=lambda x, name=None: BN_Relu(x),
                                 data_format=DATA_FORMAT,
                                 name='decoder4_3')
    #decoder4_4 = UnetUpsample(prefix='upsample4_4', l = decoder3, BASE_FILTER4)
    decoder4_4 = UnetUpsample(prefix='upsample4_4', l=decoder3, num_filters=BASE_FILTER4,scale=2)

    decoder4_4 = tf.layers.conv3d(inputs = decoder4_4,
                                 filters=BASE_FILTER4,  #80
                                 kernel_size=(3, 3, 3),
                                 strides=1,
                                 padding=PADDING,
                                 activation=lambda x, name=None: BN_Relu(x),
                                 data_format=DATA_FORMAT,
                                 name='decoder4_4')
    decoder4_5 = tf.layers.conv3d(inputs = layer1,
                                 filters=BASE_FILTER4,  #80
                                 kernel_size=(3, 3, 3),
                                 strides=1,
                                 padding=PADDING,
                                 activation=lambda x, name=None: BN_Relu(x),
                                 data_format=DATA_FORMAT,
                                 name='decoder4_5')
    decoder4 = tf.concat([decoder4_1, decoder4_2, decoder4_3, decoder4_4, decoder4_5], axis=1)
    decoder4 = tf.layers.conv3d(inputs = decoder4,
                                 filters=BASE_FILTER4,  #80
                                 kernel_size=(3, 3, 3),
                                 strides=1,
                                 padding=PADDING,
                                 activation=lambda x, name=None: BN_Relu(x),
                                 data_format = DATA_FORMAT,
                                 name='decoder4_6')
    #output_map = conv_sigmoid(x=decoder4, kernal=(1, 1, 1, 20, n_class), scope='output')

    layer = tf.layers.conv3d(decoder4,
                             filters=config.NUM_CLASS,
                             kernel_size=(1, 1, 1),
                             padding="SAME",
                             activation=tf.identity,
                             data_format=DATA_FORMAT,
                             name="final")
    layer = tf.sigmoid(layer)

    # if config.DEEP_SUPERVISION:
    #     layer = layer + deep_supervision
    if DATA_FORMAT == 'channels_first':
        layer = tf.transpose(layer, [0, 2, 3, 4, 1])  # to-channel last
    print("final", layer.shape)  # [3, num_class, d, h, w]
    return layer

# Max Pooling
def max_pool3d(inputs, stride,data_format=DATA_FORMAT ,depth=False):
    """
        depth flag:False is z axis is same between input and output,true is z axis is input is twice than output
        """
    if data_format=="channels_first":
        data_format='NCDHW'
        if depth:
            pool3d = tf.nn.max_pool3d(inputs, ksize=[1, 1,stride, stride, stride],
                                      strides=[1, 1,stride, stride, stride], padding='SAME', data_format=data_format)
        else:  # channel_first add or not?
            pool3d = tf.nn.max_pool3d(inputs, ksize=[1, 1,stride, stride, 1], strides=[1,  1,stride, stride, 1],
                                      padding='SAME', data_format=data_format)
    else:
        data_format='NDHWC'
        if depth:
            pool3d = tf.nn.max_pool3d(inputs, ksize=[1, 1, stride, stride, stride],
                                      strides=[1,  stride, stride, stride,1], padding='SAME', data_format=data_format)
        else:  # channel_first add or not?
            pool3d = tf.nn.max_pool3d(inputs, ksize=[1, 1,stride, stride, 1], strides=[1, 1, stride, stride, 1],
                                      padding='SAME', data_format=data_format)

    return pool3d
def Upsample3D(prefix, l, scale=2,data_formate = DATA_FORMAT):
    l = tf.keras.layers.UpSampling3D(size=(scale, scale, scale), data_format =  data_formate,name = prefix)(l)

    """
    l = tf.layers.conv3d_transpose(inputs=l, 
                                filters=config.NUM_CLASS,
                                kernel_size=(2,2,2),
                                strides=2,
                                padding=PADDING,
                                activation=tf.nn.relu,
                                data_format=DATA_FORMAT,
                                name="upsampe_{}".format(prefix))

    l_out = tf.identity(l)
    if DATA_FORMAT == 'channels_first':
        l = tf.transpose(l, [0, 2, 3, 4, 1])
    l_shape = l.get_shape().as_list()
    l = tf.reshape(l, [l_shape[0]*l_shape[1], l_shape[2], l_shape[3], l_shape[4]])
    l = tf.image.resize_images(l , (l_shape[2]*scale, l_shape[3]*scale))
    l = tf.reshape(l, [l_shape[0], l_shape[1], l_shape[2]*scale, l_shape[3]*scale, l_shape[4]])
    if DATA_FORMAT == 'channels_first':
        l = tf.transpose(l, [0, 4, 1, 2, 3]) # Back to channel_first
    """
    return l


def UnetUpsample(prefix, l, num_filters,data_format=DATA_FORMAT,scale=2):
    """
    l = tf.layers.conv3d_transpose(inputs=l,
                                filters=num_filters,
                                kernel_size=(2,2,2),
                                strides=2,
                                padding=PADDING,
                                activation=tf.nn.relu,
                                data_format=DATA_FORMAT,
                                name="up_conv0_{}".format(prefix))
    """
    """   
     Upsample3D(prefix, l, scale=2,data_formate = DATA_FORMAT)
     tf.keras.layers.UpSampling3D(size=(scale, scale, scale), data_format =  data_formate,name = prefix)
    """
    l = Upsample3D(prefix, l, scale,data_format)
    l = tf.layers.conv3d(inputs=l,
                         filters=num_filters,
                         kernel_size=(3, 3, 3),
                         strides=1,
                         padding=PADDING,
                         activation=lambda x, name=None: BN_Relu(x),
                         data_format=DATA_FORMAT,
                         name="up_conv_{}".format(prefix))
    return l


def BN_Relu(x):
    if config.INSTANCE_NORM:
        l = InstanceNorm5d('ins_norm', x, data_format=DATA_FORMAT)
    else:
        l = BatchNorm3d('bn', x, axis=1 if DATA_FORMAT == 'channels_first' else -1)
    l = tf.nn.relu(l)
    return l


def Unet3dBlock(prefix, l, kernels, n_feat, s):
    if config.RESIDUAL:
        l_in = l

    for i in range(2):
        l = tf.layers.conv3d(inputs=l,
                             filters=n_feat,
                             kernel_size=kernels,
                             strides=1,
                             padding=PADDING,
                             activation=lambda x, name=None: BN_Relu(x),
                             data_format=DATA_FORMAT,
                             name="{}_conv_{}".format(prefix, i))

    return l_in + l if config.RESIDUAL else l


### from niftynet ####
def labels_to_one_hot(ground_truth, num_classes=1):
    """
    Converts ground truth labels to one-hot, sparse tensors.
    Used extensively in segmentation losses.
    :param ground_truth: ground truth categorical labels (rank `N`)
    :param num_classes: A scalar defining the depth of the one hot dimension
        (see `depth` of `tf.one_hot`)
    :return: one-hot sparse tf tensor
        (rank `N+1`; new axis appended at the end)
    """
    # read input/output shapes
    if isinstance(num_classes, tf.Tensor):
        num_classes_tf = tf.to_int32(num_classes)
    else:
        num_classes_tf = tf.constant(num_classes, tf.int32)
    input_shape = tf.shape(ground_truth)
    output_shape = tf.concat(
        [input_shape, tf.reshape(num_classes_tf, (1,))], 0)

    if num_classes == 1:
        # need a sparse representation?
        return tf.reshape(ground_truth, output_shape)

    # squeeze the spatial shape
    ground_truth = tf.reshape(ground_truth, (-1,))
    # shape of squeezed output
    dense_shape = tf.stack([tf.shape(ground_truth)[0], num_classes_tf], 0)

    # create a rank-2 sparse tensor
    ground_truth = tf.to_int64(ground_truth)
    ids = tf.range(tf.to_int64(dense_shape[0]), dtype=tf.int64)
    ids = tf.stack([ids, ground_truth], axis=1)
    one_hot = tf.SparseTensor(
        indices=ids,
        values=tf.ones_like(ground_truth, dtype=tf.float32),
        dense_shape=tf.to_int64(dense_shape))

    # resume the spatial dims
    one_hot = tf.sparse_reshape(one_hot, output_shape)
    return one_hot


def generalised_dice_loss(prediction,
                          ground_truth,
                          weight_map=None,
                          type_weight='Square'):
    """
    Function to calculate the Generalised Dice Loss defined in
        Sudre, C. et. al. (2017) Generalised Dice overlap as a deep learning
        loss function for highly unbalanced segmentations. DLMIA 2017
    :param prediction: the logits
    :param ground_truth: the segmentation ground truth
    :param weight_map:
    :param type_weight: type of weighting allowed between labels (choice
        between Square (square of inverse of volume),
        Simple (inverse of volume) and Uniform (no weighting))
    :return: the loss
    """
    prediction = tf.cast(prediction, tf.float32)
    if len(ground_truth.shape) == len(prediction.shape):
        ground_truth = ground_truth[..., -1]
    one_hot = labels_to_one_hot(ground_truth, tf.shape(prediction)[-1])

    if weight_map is not None:
        n_classes = prediction.shape[1].value
        weight_map_nclasses = tf.reshape(
            tf.tile(weight_map, [n_classes]), prediction.get_shape())
        ref_vol = tf.sparse_reduce_sum(
            weight_map_nclasses * one_hot, reduction_axes=[0])

        intersect = tf.sparse_reduce_sum(
            weight_map_nclasses * one_hot * prediction, reduction_axes=[0])
        seg_vol = tf.reduce_sum(
            tf.multiply(weight_map_nclasses, prediction), 0)
    else:
        ref_vol = tf.sparse_reduce_sum(one_hot, reduction_axes=[0])
        intersect = tf.sparse_reduce_sum(one_hot * prediction,
                                         reduction_axes=[0])
        seg_vol = tf.reduce_sum(prediction, 0)
    if type_weight == 'Square':
        weights = tf.reciprocal(tf.square(ref_vol))
    elif type_weight == 'Simple':
        weights = tf.reciprocal(ref_vol)
    elif type_weight == 'Uniform':
        weights = tf.ones_like(ref_vol)
    else:
        raise ValueError("The variable type_weight \"{}\""
                         "is not defined.".format(type_weight))
    new_weights = tf.where(tf.is_inf(weights), tf.zeros_like(weights), weights)
    weights = tf.where(tf.is_inf(weights), tf.ones_like(weights) *
                       tf.reduce_max(new_weights), weights)
    generalised_dice_numerator = \
        2 * tf.reduce_sum(tf.multiply(weights, intersect))
    generalised_dice_denominator = \
        tf.reduce_sum(tf.multiply(weights, seg_vol + ref_vol)) + 1e-6
    generalised_dice_score = \
        generalised_dice_numerator / generalised_dice_denominator
    return 1 - generalised_dice_score




def dice(prediction, ground_truth, weight_map=None):
    """
    Function to calculate the dice loss with the definition given in
        Milletari, F., Navab, N., & Ahmadi, S. A. (2016)
        V-net: Fully convolutional neural
        networks for volumetric medical image segmentation. 3DV 2016
    using a square in the denominator
    :param prediction: the logits
    :param ground_truth: the segmentation ground_truth
    :param weight_map:
    :return: the loss
    """
    ground_truth = tf.to_int64(ground_truth)
    ids = tf.range(tf.to_int64(tf.shape(ground_truth)[0]), dtype=tf.int64)

    #inds_not_zero = tf.reshape(tf.where(tf.not_equal(ground_truth,tf.zeros_like(ground_truth))),[-1])
    #ground_truth2 = tf.gather(ground_truth,inds_not_zero)
    #ids2 = tf.gather(ids,inds_not_zero)
    prediction = tf.cast(prediction, tf.float32)

    ids = tf.stack([ids, ground_truth], axis=1)

    one_hot = tf.SparseTensor(
        indices=ids,
        values=tf.ones_like(ground_truth, dtype=tf.float32),
        dense_shape=tf.to_int64(tf.shape(prediction)))
    one_hot2 = tf.SparseTensor(
        indices=ids,
        values=tf.ones_like(ground_truth, dtype=tf.float32),
        dense_shape=tf.to_int64(tf.shape(prediction))
    )
    if weight_map is not None:
        n_classes = prediction.shape[1].value

        weight_prediction = tf.ones_like(prediction)
        weight_prediction_0 = weight_prediction[:,0]*1
        weight_prediction_1 = weight_prediction[:,1]*1
        weight_prediction_2 = weight_prediction[:,2]*1
        weight_prediction_3 = weight_prediction[:,3]*1

        weight_prediction_all = tf.stack([weight_prediction_0,weight_prediction_1,weight_prediction_2,weight_prediction_3],1)


        weight_map_nclasses = tf.reshape(
            tf.tile(weight_map, [n_classes]), prediction.get_shape())
        dice_numerator = 2.0 * tf.sparse_reduce_sum(
            weight_map_nclasses * one_hot * prediction*weight_prediction_all, reduction_axes=[0])
        dice_denominator = \
            tf.reduce_sum(weight_map_nclasses * tf.square(prediction),
                          reduction_indices=[0]) + \
            tf.sparse_reduce_sum(one_hot * weight_map_nclasses,
                                 reduction_axes=[0])
    else:

        dice_numerator = 2.0 * tf.sparse_reduce_sum(
            one_hot * prediction * weight_prediction, reduction_axes=[0])
        dice_denominator = \
            tf.reduce_sum(tf.square(prediction), reduction_indices=[0]) + \
            tf.sparse_reduce_sum(one_hot, reduction_axes=[0])
    epsilon_denominator = 0.00001

    dice_score = dice_numerator / (dice_denominator + epsilon_denominator)

    return 1.0 - tf.reduce_mean(dice_score)


def Loss(feature, weight, gt):
    # compute batch-wise
    losses = []
    for idx in range(config.BATCH_SIZE):
        f = tf.reshape(feature[idx], [-1, config.NUM_CLASS])

        # f = tf.cast(f, dtype=tf.float32)
        # f = tf.nn.softmax(f)

        w = tf.reshape(weight[idx], [-1])
        g = tf.reshape(gt[idx], [-1])
        print(f.shape, w.shape, g.shape)
        if g.shape.as_list()[-1] == 1:
            g = tf.squeeze(g, axis=-1)  # (nvoxel, )
        if w.shape.as_list()[-1] == 1:
            w = tf.squeeze(w, axis=-1)  # (nvoxel, )
        f = tf.nn.softmax(f)
        '''
        '''
        loss_per_batch = dice(f, g, weight_map=w)
        # loss_per_batch = cross_entropy(f, g, weight_map=w)
        losses.append(loss_per_batch)
    return tf.reduce_mean(losses, name="dice_loss")


if __name__ == "__main__":
    image = tf.transpose(tf.constant(np.zeros((config.BATCH_SIZE, 128, 128, 128, 4)).astype(np.float32)),
                         [0, 4, 1, 2, 3])
    gt = tf.constant(np.zeros((config.BATCH_SIZE, 128, 128, 128, 1)).astype(np.float32))
    weight = tf.constant(np.ones((config.BATCH_SIZE, 128, 128, 128, 1)).astype(np.float32))
    t = unet3d('unet3d', image)
    loss = Loss(t, weight, gt)
    print(t.shape, loss)