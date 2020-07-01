
from keras.layers.advanced_activations import LeakyReLU
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.layers import (
    Dense,
    Flatten, Dropout,
    BatchNormalization, Activation,
    Lambda, Layer, Add, Concatenate,
    Average,GaussianNoise,
    MaxPooling2D, AveragePooling2D,
    GlobalAveragePooling2D,
)
from keras.layers.convolutional import (
    UpSampling2D, Convolution2D,
    Conv2D, Conv2DTranspose
)
# =================== Custom layers ====================#

class SelfAttention(Layer):
    def __init__(self, ch, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)
        self.channels = ch
        self.filters_f_g = self.channels // 8
        self.filters_h = self.channels

    def build(self, input_shape):
        kernel_shape_f_g = (1, 1) + (self.channels, self.filters_f_g)
        kernel_shape_h = (1, 1) + (self.channels, self.filters_h)

        # Create a trainable weight variable for this layer:
        self.gamma = self.add_weight(name='gamma', shape=[1], initializer='zeros', trainable=True)
        self.kernel_f = self.add_weight(shape=kernel_shape_f_g,
                                        initializer='glorot_uniform',
                                        name='kernel_f',
                                        trainable=True)
        self.kernel_g = self.add_weight(shape=kernel_shape_f_g,
                                        initializer='glorot_uniform',
                                        name='kernel_g',
                                        trainable=True)
        self.kernel_h = self.add_weight(shape=kernel_shape_h,
                                        initializer='glorot_uniform',
                                        name='kernel_h',
                                        trainable=True)

        super(SelfAttention, self).build(input_shape)
        # Set input spec.
        self.input_spec = keras.layers.InputSpec(ndim=4,
                                    axes={3: input_shape[-1]})
        self.built = True

    def call(self, x):
        def hw_flatten(x):
            return K.reshape(x, shape=[K.shape(x)[0], K.shape(x)[1]*K.shape(x)[2], K.shape(x)[3]])

        f = K.conv2d(x,
                     kernel=self.kernel_f,
                     strides=(1, 1), padding='same')  # [bs, h, w, c']
        g = K.conv2d(x,
                     kernel=self.kernel_g,
                     strides=(1, 1), padding='same')  # [bs, h, w, c']
        h = K.conv2d(x,
                     kernel=self.kernel_h,
                     strides=(1, 1), padding='same')  # [bs, h, w, c]

        s = K.batch_dot(hw_flatten(g), K.permute_dimensions(hw_flatten(f), (0, 2, 1)))  # # [bs, N, N]

        beta = K.softmax(s, axis=-1)  # attention map

        o = K.batch_dot(beta, hw_flatten(h))  # [bs, N, C]

        o = K.reshape(o, shape=K.shape(x))  # [bs, h, w, C]
        x = self.gamma * o + x

        return x

    def compute_output_shape(self, input_shape):
        return input_shape


class FeatureNorm(keras.layers.Layer):
    def __init__(self, epsilon = 1e-4, norm = 'bn'):
        super(FeatureNorm, self).__init__()
        self.epsilon = epsilon
        self.norm = norm

    def call(self, inputs):
        x, scale, bias = inputs
        # x = [batch, height, width, channels]
        N, H, W, C = x.shape

        if 'bn' in self.norm:
            logger.info('Use Batch norm for FeatureNorm layer')
            axis = [0, 1, 2]
        else:
            # instance norm
            logger.info('Use Instance norm for FeatureNorm layer')
            axis = [1, 2]

        mean = K.mean(x, axis = axis, keepdims = True)
        std = K.std(x, axis = axis, keepdims = True)
        norm = (x - mean) * (1 / (std + self.epsilon))

        broadcast_scale = K.reshape(scale, (-1, 1, 1, C))
        broadcast_bias = K.reshape(bias, (-1, 1, 1, C))

        return norm * broadcast_scale + broadcast_bias

    def compute_output_shape(self, input_shape):
        return input_shape[0]

def down_sample(x, scale_factor_h, scale_factor_w) :
    _, h, w, _ = x.get_shape().as_list()
    new_size = [h // scale_factor_h, w // scale_factor_w]

    return tf.image.resize_nearest_neighbor(x, size=new_size)
class Spade(keras.layers.Layer):
    def __init__(self, channels):
        super(Spade, self).__init__()
        self.channels = channels
        self.epsilon = 1e-4

    def call(self, inputs):
        x, image = inputs
        # x = [batch, height, width, channels]
        x_n, x_h, x_w, x_c = K.int_shape(x)
        _, i_h, i_w, _ = K.int_shape(image)

        factor_h = i_h // x_h  # 256 // 4 = 64
        factor_w = i_w // x_w

        image_down = Lambda(lambda x: down_sample(x, factor_h, factor_w))(image)
        image_down = Conv2D(128, kernel_size=5, strides=1,
                            padding='same',
                            activation='relu')(image_down)
        
        image_gamma = Conv2D(self.channels, kernel_size=5,
                            strides=1, padding='same')(image_down)
        image_beta = Conv2D(self.channels, kernel_size=5,
                            strides=1, padding='same')(image_down)

        # axis = [0, 1, 2] # batch
        # mean = K.mean(x, axis = axis, keepdims = True)
        # std = K.std(x, axis = axis, keepdims = True)
        # norm = (x - mean) * (1 / (std + self.epsilon))

        return x * (1 + image_beta) + image_gamma

    def compute_output_shape(self, input_shape):
        return input_shape[0]

# ==================== Functions ======================= #
def actv(activation):
    if activation == 'leaky_relu':
        return LeakyReLU()
    return Activation(activation)


def apply_norm(x, norm='batch', norm_func):
    if imgs is not None:
        x = self._apply_feature_norm(x, imgs)
    elif batch in norm:
        x = BatchNormalization()(x)
    elif 'in' in norm:
        x = InstanceNormalization()(x)
    return x


def up_resblock(x,
                units = 64,
                kernel_size = 3,
                activation = 'leaky_relu',
                norm = 'batch',
                attr_image=None):

    interpolation = 'nearest'

    out = apply_norm(x, imgs=attr_image)
    out = actv(activation)(out)
    out = UpSampling2D(size=(2, 2), interpolation=interpolation)(out)
    out = Conv2D(units, kernel_size, strides = 1, padding='same')(out)

    out = apply_norm(out, imgs=attr_image)
    out = actv(activation)(out)
    out = Conv2D(units, kernel_size, strides = 1, padding='same')(out)

    x = UpSampling2D(size=(2, 2), interpolation=interpolation)(x)
    x = Conv2D(units, kernel_size, strides = 1, padding='same')(x)

    return Add()([out, x])


def down_resblock(x):
    pass