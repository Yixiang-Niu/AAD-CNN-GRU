import tensorflow as tf

def AAD(shape_eeg, shape_sti,
        kernels=32, kernel_size=7, strides=1, dilation_rate=3,
        units=48):
    """ Parameters:
        shape_eeg:     tuple, shape of EEG (time, channel)
        shape_sti:     tuple, shape of stimulus (time, feature_dim)
        kernels:       int, number of output filters in the 1D convolution
        kernel_size:   int, length of the 1D convolution window
        strides:       int, stride length of the 1D convolution
        dilation_rate: int, dilation rate to use for the dilated 1D convolution
        units:         int, dimensionality of the output space of the GRU layer
    """
    # Inputs
    input0 = tf.keras.layers.Input(shape=shape_eeg);    eeg = input0 # EEG or CSP-filtered EEG
    input1 = tf.keras.layers.Input(shape=shape_sti);    sti1 = input1 # 1st speech within a mixed stimulus
    input2 = tf.keras.layers.Input(shape=shape_sti);    sti2 = input2 # 2nd speech

    # Path for EEG
    eeg = tf.keras.layers.Conv1D(kernels, kernel_size, strides=strides, dilation_rate=dilation_rate, activation=None)(eeg)
    eeg = tf.keras.layers.BatchNormalization()(eeg)
    eeg = tf.compat.v1.keras.layers.CuDNNGRU(units, return_sequences=True)(eeg)

    # Path for stimulus
    Conv1D = tf.keras.layers.Conv1D(kernels, kernel_size, strides=strides, dilation_rate=dilation_rate, activation=None)
    sti1 = Conv1D(sti1)
    sti2 = Conv1D(sti2)

    BN = tf.keras.layers.BatchNormalization()
    sti1 = BN(sti1)
    sti2 = BN(sti2)

    GRU = tf.compat.v1.keras.layers.CuDNNGRU(units, return_sequences=True)
    sti1 = GRU(sti1)
    sti2 = GRU(sti2)

    # Similarity measure
    Dot1 = tf.keras.layers.dot([eeg, sti1], 2, normalize=True)
    CosSim1 = tf.math.reduce_mean(tf.linalg.diag_part(Dot1), axis=-1)
    CosSim1 = tf.expand_dims(CosSim1, axis=-1)
    Dot2 = tf.keras.layers.dot([eeg, sti2], 2, normalize=True)
    CosSim2 = tf.math.reduce_mean(tf.linalg.diag_part(Dot2), axis=-1)
    CosSim2 = tf.expand_dims(CosSim2, axis=-1)

    CosSim_concat = tf.keras.layers.concatenate([CosSim1, CosSim2])

    one_hot = tf.keras.layers.Dense(2, activation='softmax')(CosSim_concat)

    # Building a model
    model = tf.keras.Model(inputs=[input0, input1, input2], outputs=one_hot)
    model.summary()
    model.compile(loss='categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  metrics=['categorical_accuracy'])
    return model