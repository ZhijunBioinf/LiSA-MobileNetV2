import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras import backend as K


def se_module(x, reduction=16):
    """
    Squeeze-and-Excitation (SE) attention mechanism
    :param x: the feature map
    :param reduction: ratio of dimension reduction (default: 16)
    :return: the feature map after involving SE mechanism
    """
    channels = K.int_shape(x)[-1]  
    
 
    squeeze = layers.GlobalAveragePooling2D()(x)  # shape: (batch_size, channels)
    squeeze = layers.Reshape((1, 1, channels))(squeeze)  # shape: (batch_size, 1, 1, channels)
    
    # **Excitation：
    excitation = layers.Dense(channels // reduction, activation="swish")(squeeze)  
    excitation = layers.Dense(channels, activation="sigmoid")(excitation)  
    
   
    x = layers.Multiply()([x, excitation])
    
    return x


def inverted_residual_block(x, filters, stride=1, expansion=4, use_se=True):
    shortcut = x
    in_channels = K.int_shape(x)[-1]


    x = layers.Conv2D(expansion * in_channels, 1, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = tf.keras.activations.swish(x)

    x = layers.DepthwiseConv2D(3, strides=stride, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = tf.keras.activations.swish(x)


    x = layers.Conv2D(filters, 1, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    if use_se:
        x = se_module(x)


    if stride == 1 and in_channels == filters:
        x = layers.Add()([x, shortcut])

    return x

def OptimizedMobileNetV2(input_shape=(224, 224, 3), num_classes=10, alpha=1.0):
    inputs = tf.keras.Input(shape=input_shape)


    x = layers.Conv2D(int(16 * alpha), 3, strides=2, padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = tf.keras.activations.swish(x)

    blocks = [
        (16, 1, 2, 2, False),  # expansion=2, do not use SE
        (24, 2, 3, 3, False),  # expansion=3, do not use SE
        (32, 2, 4, 3, False),   # expansion=4, use SE
        (64, 2, 6, 3, True),   # expansion=6, use SE
        (96, 2, 8, 2, True),   # expansion=6, use SE
    ]

    for filters, stride, expansion, repeat, use_se in blocks:
        for i in range(repeat):
            current_stride = stride if i == 0 else 1
            x = inverted_residual_block(x, int(filters * alpha), current_stride, expansion, use_se)

    x = layers.Conv2D(int(1280 * alpha), 1, use_bias=False, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = tf.keras.activations.swish(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    return Model(inputs, outputs)

model = OptimizedMobileNetV2(alpha=1.0)
model.summary()
