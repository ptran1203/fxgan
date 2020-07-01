def build_encode_decode_G(self):
    def _transpose_block(x, units, activation, kernel_size=3, norm='batch',image=None):
        def _norm_layer(x):
            if 'batch' in norm:
                x = BatchNormalization()(x)
            elif 'in' in norm:
                x = InstanceNormalization()(x)
            else:
                x = Spade(units)([x, image])
            return x

        out = Conv2DTranspose(units, kernel_size, strides=2, padding='same')(x)
        out = _norm_layer(out)
        out = activation(out)
        out = Dropout(0.3)(out)
        return out

    images = Input(shape=(self.k_shot, self.resolution, self.resolution, 3), name = 'G_input')
    latent_code = Input(shape=(self.latent_size,), name = 'latent_code')
    image = Lambda(lambda x: x[:, 0,])(images)
    attr_features = []
    for i in range(self.k_shot):
        attr_features.append(self.latent_encoder(
            Lambda(lambda x: x[:, i,])(images)
        ))

    latent_from_i = Average()(attr_features) # vector 128
    # concatenate attribute feature and latent code
    latent_from_i = Concatenate()([latent_from_i, latent_code])

    kernel_size = 3
    init_channels = 256
    norm = 'fn' if 'fn' in self.norm else self.norm

    latent = Dense(4 * 4 * init_channels)(latent_from_i)
    latent = self._norm()(latent)
    latent = Activation('relu')(latent)
    latent = Reshape((4, 4, init_channels))(latent)

    de = _transpose_block(latent, 256, Activation('relu'),
                            kernel_size, norm=norm,
                            image=image)

    if self.attention:
        de = SelfAttention(256)(de)

    de = _transpose_block(de, 128, Activation('relu'),
                            kernel_size, norm=norm,
                            image=image)

    de = _transpose_block(de, 64, Activation('relu'),
                            kernel_size, norm=norm,
                            image=image)

    final = Conv2DTranspose(self.channels, kernel_size, strides=2, padding='same')(de)
    outputs = Activation('tanh')(final)

    self.generator = Model(
        inputs = [images, latent_code],
        outputs = outputs,
        name='dc_gen',
    )