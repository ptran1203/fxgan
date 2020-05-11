class FeatureGan:
    def build_fgenerator(self):
        """
        input = latent code 100
        output 1: 32 * 32 * 64 = 65536
        output 2: 16 * 16 * 64 = 16384
        output 3: 4 * 4 * 128 = 2048
        """

        latent_code = Input(shape=(100,))

        x = Dense(256, activation = 'relu')(latent_code)
        x = Dropout(0.3)(x)
        x = Dense(512, activation = 'relu')(x)
        x = Dropout(0.3)(x)
        x = Dense(1024, activation = 'relu')(x)
        x = Dropout(0.3)(x)
        x1 = Dense(4 * 4 * 128)

        # 4 * 4 * 128
        x1 = Reshape((4, 4, 128), activation = 'tanh')(x1)

        # 8 * 8 * 128
        x2 = Conv2DTranspose(128, 3, strides = 2, padding = 'same')(x1)

        # 16 * 16 * 64
        x2 = Conv2DTranspose(64, 3, strides = 2, padding = 'same')(x1)

        # 32 * 32 * 64
        x3 = Conv2DTranspose(64, 3, strides = 2, padding = 'same')(x2)

        self.f_generator = Model(
            inputs = latent_code,
            # outputs = [x1 ,x2, x3]
            outputs = x1,
            name = 'Feature_generator'
        )

    def build_fdiscriminator(self):
        # feature_3 = Input(shape = (32, 32, 64), name = 'feature_input_3')
        # feature_2 = Input(shape = (16, 16, 64), name = 'feature_input_2')
        feature_1 = Input(shape = (4, 4, 128), name = 'feature_input_1')

        x = Flatten()(feature_1)
        x = Dense(1024, activation = 'relu')(x)
        x = Dropout(0.3)(x)
        x = Dense(512, activation = 'relu')(x)
        x = Dropout(0.3)(x)
        x = Dense(256, activation = 'relu')(x)
        x = Dropout(0.3)(x)

        outputs = Activation('sigmoid')(x)

        self.f_discriminator = Model(
            inputs = feature_1,
            outputs = outputs,
            name = 'Feature_discriminator',
        )

    def __init__(self):
        self.train_history = defaultdict(list)
        self.test_history = defaultdict(list)

        self.build_fgenerator()
        self.build_fdiscriminator()


        self.f_discriminator.compile(
            optimizer = Adam(0.001),
            metrics = ['accuracy'],
            loss = 'binary_crossentropy',
        )

        latent_code = Input(shape=(100,))

        fake_feature = self.f_generator(latent_code)

        self.f_discriminator.trainable = False
        self.f_generator.trainable = True

        aux = self.f_discriminator(fake_feature)

        self.f_combined = Model(
            inputs = latent_code,
            outputs = aux,
        )

        self.f_combined.compile(
            optimizer = Adam(0.001),
            metrics = ['acc'],
            loss = 'binary_crossentropy'
        )

    def f_train(self, epochs):
        for e in range(epochs):
            start_time = datetime.datetime.now()
            print('Feature GAN train epoch: {}/{}'.format(e+1, epochs))
            train_disc_loss, train_gen_loss, train_disc_acc, train_gen_acc = self.f_train_one_epoch(bg_train)

            
            self.train_history['disc_loss'].append(train_disc_loss)
            self.train_history['gen_loss'].append(train_gen_loss)
            # self.test_history['disc_loss'].append(test_disc_loss)
            # self.test_history['gen_loss'].append(test_gen_loss)
            # accuracy
            self.train_history['disc_acc'].append(train_disc_acc)
            self.train_history['gen_acc'].append(train_gen_acc)
            # self.test_history['disc_acc'].append(test_disc_acc)
            # self.test_history['gen_acc'].append(test_gen_acc)

            print("D_loss {}, G_loss {}, D_acc {}, G_acc {} - {}".format(
                train_disc_loss, train_gen_loss, train_disc_acc, train_gen_acc,
                datetime.datetime.now() - start_time
            ))

    def f_generate_latent(self, c):
        return np.array([
            np.random.normal(0, 1, 100)
            for e in c
        ])

    def f_train_one_epoch(self, bg_train):
        epoch_disc_loss = []
        epoch_gen_loss = []
        epoch_disc_acc = []
        epoch_gen_acc = []

        for image_batch, label_batch in bg_train.next_batch():
            crt_batch_size = label_batch.shape[0]

            ################## Train Discriminator ##################\

            f = self.generate_features(
                                self._biased_sample_labels(crt_batch_size),
                                from_p = from_p
                            )
            
            img_1 = bg_train.get_samples_for_class(0, crt_batch_size // 4)
            img_2 =  bg_train.get_samples_for_class(1, crt_batch_size // 4)

            generated_images = self.generator.predict(
                [
                    np.concatenate([img_1, img_2]),
                    f,
                ],
                verbose=0
            )
    
            X = np.concatenate((image_batch, generated_images))
            aux_y = np.concatenate((label_batch, np.full(generated_images.shape[0] , self.nclasses )), axis=0)
            
            X, aux_y = self.shuffle_data(X, aux_y)
            loss, acc = self.discriminator.train_on_batch(X, aux_y)
            epoch_disc_loss.append(loss)
            epoch_disc_acc.append(acc)

            ################## Train Generator ##################

            shuffle_image_batch = bg_train.get_samples_by_labels(label_batch)
            real_features, perceptual_features = self.get_pair_features(shuffle_image_batch)

            f = self.generate_features(
                                self._biased_sample_labels(crt_batch_size),
                                from_p = from_p
                            )

            [loss, acc] = self.f_combined.train_on_batch(
                [image_batch, f],
                [label_batch, real_features, perceptual_features, ]
            )

            epoch_gen_loss.append(loss)
            epoch_gen_acc.append(acc)

        return (
            np.mean(np.array(epoch_disc_loss), axis=0),
            np.mean(np.array(epoch_gen_loss), axis=0),
            np.mean(np.array(epoch_disc_acc), axis=0),
            np.mean(np.array(epoch_gen_acc), axis=0),
        )
