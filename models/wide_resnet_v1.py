import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Conv2D, Dropout, Flatten, Dense, AveragePooling2D


def residual_block(x, block_id, is_training, in_nb_filters=16, nb_filters=16, stride=1, drop_prob=0.3, use_conv=False):
    shortcut = x
    strides = (stride, stride)
    prefix = 'resblock_%d/' % block_id
    if nb_filters != in_nb_filters:
        shortcut = AveragePooling2D(pool_size=strides, strides=strides, padding='valid', name=prefix+'pool0')(x)
        shortcut = tf.pad(shortcut, [[0, 0], [0, 0], [0, 0],
                          [(nb_filters-in_nb_filters)//2, (nb_filters-in_nb_filters)//2]])
    x = BatchNormalization(name=prefix+'batch1')(x, training=is_training)
    x = tf.nn.relu(x, name=prefix+'relu1')
    x = Conv2D(nb_filters, (3, 3), strides=strides, padding='same', name=prefix+'conv1')(x)
    x = BatchNormalization(name=prefix+'batch2')(x, training=is_training)
    x = tf.nn.relu(x, name=prefix+'relu2')
    x = Dropout(rate=drop_prob, name=prefix+'dropout1')(x, training=is_training)
    x = Conv2D(nb_filters, (3, 3), strides=(1, 1), padding='same', name=prefix+'conv2')(x)
    x = tf.add(x, shortcut)
    return x


def wide_residual_network(x, is_training, params):
    assert 'depth' in params, 'depth must in params'
    assert 'width' in params, 'width must in params'
    assert 'drop_prob' in params, 'drop_prob must in params'
    assert 'out_units' in params, 'out_units must in params'

    depth = params['depth']
    width = params['width']
    drop_prob = params['drop_prob']
    # if use_conv, a 1*1 conv2d will be used for downsampling between groups
    if 'use_conv' in params:
        use_conv = params['use_conv']
    else:
        use_conv = False
    assert (depth - 4) % 6 == 0
    num_residual_units = (depth - 4) // 6
    nb_filters = [x * width for x in [16, 32, 64, 128]]
    prefix = 'main/'
    x = Conv2D(16, 3, strides=(1, 1), padding='same', name=prefix+'conv')(x)
    in_nb_filters = 16
    for i in range(0, num_residual_units):
        x = residual_block(x, is_training=is_training, in_nb_filters=in_nb_filters, nb_filters=nb_filters[0],
                           block_id=1, stride=1, drop_prob=drop_prob, use_conv=False)
        in_nb_filters = nb_filters[0]
        
    for i in range(0, num_residual_units):
        stride = 2 if i == 0 else 1
        x = residual_block(x, is_training=is_training, in_nb_filters=in_nb_filters, nb_filters=nb_filters[1],
                           block_id=2, stride=stride, drop_prob=drop_prob, use_conv=use_conv)
        in_nb_filters = nb_filters[1]
    for i in range(0, num_residual_units):
        stride = 2 if i == 0 else 1
        x = residual_block(x, is_training=is_training, in_nb_filters=in_nb_filters, nb_filters=nb_filters[2],
                           block_id=3, stride=stride, drop_prob=drop_prob, use_conv=use_conv)
        in_nb_filters = nb_filters[2]
    for i in range(0, num_residual_units):
        stride = 2 if i == 0 else 1
        x = residual_block(x, is_training=is_training, in_nb_filters=in_nb_filters, nb_filters=nb_filters[3],
                           block_id=4, stride=stride, drop_prob=drop_prob, use_conv=use_conv)
        in_nb_filters = nb_filters[3]
    # x = Conv2D(512, 3, strides=(2, 2), padding='same', name=prefix+'conv_1')(x)
    x = BatchNormalization(name=prefix+'bn')(x, training=is_training)
    x = tf.nn.relu(x, name=prefix+'relu')
    x = AveragePooling2D(pool_size=(8, 8), strides=(1, 1), padding='valid', name=prefix+'pool')(x)
    x = Flatten(name=prefix+'flatten')(x)
    x = Dense(params['out_units'], name=prefix+'fc')(x)
    x = BatchNormalization(name=prefix + 'bn_1')(x, training=is_training)
    out = tf.math.l2_normalize(x)
    return out


def build_wide_resnet(params, has_top=False):
    input_layer = tf.keras.Input(params['input_shape'])
    pred = wide_residual_network(input_layer, False, params)
    model = tf.keras.Model(input_layer, pred)
    if has_top:
        return model
    else:
        pred = model.get_layer('main/pool').output
    model_no_top = tf.keras.Model(input_layer, pred)
    return model_no_top


if __name__ == '__main__':
    params = {
        'input_shape': (80, 80, 3),
        'depth': 10,
        'width': 2,
        'drop_prob': 0.2,
        'out_units': 128
    }
    # input_layer = tf.keras.Input(params['input_shape'])
    # pred = wide_residual_network(input_layer, False, params)
    # model = tf.keras.Model(input_layer, pred)
    # model.summary()
    model = build_wide_resnet(params, has_top=True)
    model.summary()
