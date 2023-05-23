import tensorflow as tf

class NonNegative(tf.keras.constraints.Constraint):
    def __call__(self, w):
        return w * tf.cast(tf.math.greater_equal(w, 0.), w.dtype)

def AAD(shape_eeg, shape_sti,
        kernels=32, kernel_size=7, strides=1, dilation_rate=3,
        units=48,
        size=16, sigma=0.7,
        sources=2):
    """ Parameters:
        shape_eeg:     tuple, shape of EEG (time, channel)
        shape_sti:     tuple, shape of stimulus (time, feature_dim) or (time, feature_dim, feature_num)
        kernels:       int, number of output filters in the 1D convolution
        kernel_size:   int, length of the 1D convolution window
        strides:       int, stride length of the 1D convolution
        dilation_rate: int, dilation rate to use for the dilated 1D convolution
        units:         int, dimensionality of the output space of the GRU layer
        size:          int, size of gaussian filter of the SSIM index
        sigma:         int, width of gaussian filter of the SSIM index
        sources:       int, number of sound sources in a mixed stimulus
    """
    # Inputs
    input0 = tf.keras.layers.Input(shape=shape_eeg);    eeg = input0 # EEG or CSP-filtered EEG
    input1 = tf.keras.layers.Input(shape=shape_sti);    sti1 = input1 # feature of the 1st sound source within a mixed stimulus
    input2 = tf.keras.layers.Input(shape=shape_sti);    sti2 = input2 # 2nd sound source
    if sources == 3:
        input3 = tf.keras.layers.Input(shape=shape_sti);    sti3 = input3 # 3rd sound source

    # Path for EEG
    eeg = tf.keras.layers.Conv1D(kernels, kernel_size, strides=strides, dilation_rate=dilation_rate, activation=None)(eeg)
    eeg = tf.keras.layers.BatchNormalization()(eeg)
    # eeg = tf.keras.layers.AveragePooling1D(pool_size=2, strides=2)(eeg) # no pooling, unless out of GPU memory
    eeg = tf.compat.v1.keras.layers.CuDNNGRU(units, return_sequences=True)(eeg)

    # Path for stimulus
    if len(sti1.shape) == 4:    # the 4th value of the stimulus shape is feature_num
        feature_fusion = tf.keras.layers.Dense(1, activation=None, use_bias=False, kernel_constraint=NonNegative())
        sti1 = feature_fusion(sti1);    sti1 = tf.squeeze(sti1, -1)
        sti2 = feature_fusion(sti2);    sti2 = tf.squeeze(sti2, -1)
        if sources == 3:
            sti3 = feature_fusion(sti3);    sti3 = tf.squeeze(sti3, -1)

    Conv1D = tf.keras.layers.Conv1D(kernels, kernel_size, strides=strides, dilation_rate=dilation_rate, activation=None)
    sti1 = Conv1D(sti1)
    sti2 = Conv1D(sti2)
    if sources == 3:
        sti3 = Conv1D(sti3)

    BN = tf.keras.layers.BatchNormalization()
    sti1 = BN(sti1)
    sti2 = BN(sti2)
    if sources == 3:
        sti3 = BN(sti3)
        
    # Pooling = tf.keras.layers.AveragePooling1D(pool_size=2, strides=2) # same as above
    # sti1 = Pooling(sti1)
    # sti2 = Pooling(sti2)
    # if sources == 3:
    #     sti3 = Pooling(sti3)

    GRU = tf.compat.v1.keras.layers.CuDNNGRU(units, return_sequences=True)
    sti1 = GRU(sti1)
    sti2 = GRU(sti2)
    if sources == 3:
        sti3 = GRU(sti3)

    # Classification
    eeg = tf.expand_dims(eeg, -1);  sti1 = tf.expand_dims(sti1, -1);    sti2 = tf.expand_dims(sti2, -1)
    SSIM1 = tf.image.ssim(eeg, sti1, max_val=2, filter_size=size, filter_sigma=sigma)
    SSIM1 = tf.expand_dims(SSIM1, axis=-1)
    SSIM2 = tf.image.ssim(eeg, sti2, max_val=2, filter_size=size, filter_sigma=sigma)
    SSIM2 = tf.expand_dims(SSIM2, axis=-1)
    if sources == 3:
        sti3 = tf.expand_dims(sti3, -1)
        SSIM3 = tf.image.ssim(eeg, sti3, max_val=2, filter_size=size, filter_sigma=sigma)
        SSIM3 = tf.expand_dims(SSIM3, axis=-1)

    SSIM_concat = tf.keras.layers.concatenate([SSIM1, SSIM2])
    if sources == 3:
        SSIM_concat = tf.keras.layers.concatenate([SSIM_concat, SSIM3])

    one_hot = tf.keras.layers.Dense(units=sources, activation='softmax')(SSIM_concat)
    
    # Building a model
    if sources == 2:
        model = tf.keras.Model(inputs=[input0, input1, input2], outputs=one_hot)
    elif sources == 3:
        model = tf.keras.Model(inputs=[input0, input1, input2, input3], outputs=one_hot)
    model.summary()
    model.compile(loss='categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  metrics=['categorical_accuracy'])
    return model
