import tensorflow as tf
from tensorflow.keras import Sequential

class KernelPredict(tf.keras.layers.Layer):
    def __init__(self, in_channels, kernel_size=3, conv=False, group_div=1, name='KernelPredict'):
        super(KernelPredict, self).__init__(name=name)
        self.channels = in_channels # content feature map channels
        self.kernel_size = kernel_size
        self.group_div = group_div
        self.conv = conv
        self.n_groups = self.channels // self.group_div

        if self.conv:
            self.w_spatial_layer = tf.keras.layers.Conv2D(filters=self.channels * self.channels // self.n_groups, kernel_size=self.kernel_size,
                                                          strides=1, use_bias=True, padding='SAME', name='w_spatial_conv')
            self.w_point_layer = Sequential([tf.keras.layers.GlobalAvgPool2D(name='gap_point_conv'),
                                             tf.keras.layers.Dense(units=self.channels * self.channels // self.n_groups,
                                                                   use_bias=True, name='w_point_fc')])
            self.bias = Sequential([tf.keras.layers.GlobalAvgPool2D(name='gap_point_bias'),
                                    tf.keras.layers.Dense(units=self.channels,
                                                          use_bias=True, name='bias_fc')])
        else: # fully-connected
            self.w_spatial_layer = tf.keras.layers.Dense(units=self.channels * self.channels // self.n_groups,
                                                         use_bias=True, name='w_spatial_fc')
            self.w_point_layer = tf.keras.layers.Dense(units=self.channels * self.channels // self.n_groups,
                                                       use_bias=True, name='w_point_fc')
            self.bias = tf.keras.layers.Dense(units=self.channels,
                                              use_bias=True, name='bias_fc')

    def call(self, style_w, training=None, mask=None):
        batch_size = style_w.shape[0]
        style_w_size = style_w.shape[1]

        w_spatial = self.w_spatial_layer(style_w)

        if self.conv:
            w_spatial = tf.reshape(w_spatial, shape=[batch_size, style_w_size, style_w_size, self.channels // self.n_groups, self.channels])
        else:
            w_spatial = tf.reshape(w_spatial, shape=[batch_size, 1, 1, self.channels // self.n_groups, self.channels]) # in, out

        w_pointwise = self.w_point_layer(style_w)
        w_pointwise = tf.reshape(w_pointwise, shape=[batch_size, 1, 1, self.channels // self.n_groups, self.channels])

        bias = self.bias(style_w)
        bias = tf.reshape(bias, shape=[batch_size, self.channels])

        return w_spatial, w_pointwise, bias

class AdaConv(tf.keras.layers.Layer):
    def __init__(self, channels, kernel_size=3, group_div=1, name='AdaConv'):
        super(AdaConv, self).__init__(name=name)

        self.channels = channels
        self.kernel_size = kernel_size
        self.group_div = group_div

        self.conv = tf.keras.layers.Conv2D(filters=self.channels, kernel_size=self.kernel_size,
                                           strides=1, use_bias=True, padding='SAME', name='conv1')

    def build(self, input_shape):
        self.n_groups = input_shape[0][-1] // self.group_div

    def call(self, inputs, training=None, mask=None):
        """
        x = [batch, height, width, channels]
        w_spatial = [batch, ws_height, ws_width, in_channels, out_channels]
        w_pointwise = [batch, wp_height, wp_width, in_channels, out_channels]
        bias = [batch, out_channels]
        """

        x, w_spatial, w_pointwise, bias = inputs
        batch_size = x.shape[0]
        xs = []

        x = self._normalize(x)

        for i in range(batch_size):
            _x = self._apply_weights(x[i:i+1], w_spatial[i:i+1], w_pointwise[i:i+1], bias[i:i+1])
            xs.append(_x)

        x = tf.concat(xs, axis=0)
        x = self.conv(x)

        return x

    def _normalize(self, x, eps=1e-5):
        mean = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
        std = tf.math.reduce_std(x, axis=[1, 2], keepdims=True)
        x_norm = (x - mean) / (std + eps)

        return x_norm

    def _apply_weights(self, x, w_spatial, w_pointwise, bias):
        """
        x = [1, height, width, channels]
        w_spatial = [1, ws_height, ws_width, in_channels, out_channels]
        w_pointwise = [1, wp_height, wp_width, in_channels, out_channels]
        bias = [1, out_channels]
        """

        # spatial conv
        spatial_out_channels = w_spatial.shape[-1]
        spatial_kernel_size = w_spatial.shape[1]
        spatial_conv = tf.keras.layers.Conv2D(filters=spatial_out_channels, kernel_size=spatial_kernel_size,
                                              strides=1, use_bias=False, padding='SAME', groups=self.n_groups, name='spatial_conv')

        spatial_conv.build(x.shape)
        spatial_conv.set_weights(w_spatial)
        x = spatial_conv(x)

        # pointwise conv
        point_out_channels = w_pointwise.shape[-1]
        point_kernel_size = w_pointwise.shape[1]
        w_pointwise = tf.squeeze(w_pointwise, axis=0)
        bias = tf.squeeze(bias, axis=0)

        point_conv = tf.keras.layers.Conv2D(filters=point_out_channels, kernel_size=point_kernel_size,
                                            strides=1, use_bias=True, padding='VALID', groups=self.n_groups, name='point_conv')
        point_conv.build(x.shape)
        point_conv.set_weights([w_pointwise, bias])
        x = point_conv(x)

        return x


# test code
feats = tf.random.normal(shape=[5, 64, 64, 256])
style_w = tf.random.normal(shape=[5, 512])

kp = KernelPredict(in_channels=feats.shape[-1], group_div=1)
adac = AdaConv(channels=1024, group_div=1)

w_spatial, w_pointwise, bias = kp(style_w)
x = adac([feats, w_spatial, w_pointwise, bias])
print(x.shape)

