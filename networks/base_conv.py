import tensorflow as tf


def build_conv_block(shape, gap=True):# {{{

    inputs = tf.keras.layers.Input(shape=shape)
    x = tf.keras.layers.Conv1D(16, 3, padding='same', activation='linear', kernel_initializer='he_normal')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv1D(32, 3, padding='same', activation='linear', kernel_initializer='he_normal')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv1D(64, 3, padding='same', activation='linear', kernel_initializer='he_normal')(x)
    root = tf.keras.layers.BatchNormalization()(x)

    # block1
    x = tf.keras.layers.Conv1D(16, 1, padding='same', kernel_initializer='he_normal')(root)
    x1 = tf.keras.layers.Conv1D(128, 3, padding='same', kernel_initializer='he_normal')(x)
    # block2
    x = tf.keras.layers.Conv1D(16, 1, padding='same', kernel_initializer='he_normal')(root)
    x2 = tf.keras.layers.Conv1D(128, 5, padding='same', kernel_initializer='he_normal')(x)
    # block3
    x = tf.keras.layers.Conv1D(16, 1, padding='same', kernel_initializer='he_normal')(root)
    x3 = tf.keras.layers.Conv1D(128, 7, padding='same', kernel_initializer='he_normal')(x)

    ori = tf.keras.layers.Conv1D(128, 1, padding='same', kernel_initializer='he_normal')(root)

    x = tf.keras.layers.Add()([ori, x1, x2, x3])
    x = tf.keras.layers.Conv1D(128, 3, padding='same', activation='relu', kernel_initializer='he_normal')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    if gap:
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
    model = tf.keras.Model(inputs=inputs, outputs=x)
    model.summary()
    return model
# }}}

def build_base(embed_dim = 4):# {{{

    inputs_pi = tf.keras.layers.Input(shape=(21,))
    inputs_m = tf.keras.layers.Input(shape=(31,))
    embedding = tf.keras.layers.Embedding(input_dim=4, output_dim=embed_dim)

    # merge_input = tf.concat([inputs_pi, inputs_m], axis=1)
    # x = tf.keras.layers.Conv1D(64, 3, padding='same', activation='relu')(merge_input)
    # se = tf.keras.layers.GlobalAveragePooling1D()(x)
    # se = tf.keras.layers.Dense(64, activation='relu')(se)
    # x = tf.keras.layers.Multiply()([x, se])
    # x = tf.keras.layers.Conv1D(128, 3, padding='same', activation='relu')(x)
    # se = tf.keras.layers.GlobalAveragePooling1D()(x)
    # se = tf.keras.layers.Dense(128, activation='relu')(se)
    # x = tf.keras.layers.Multiply()([x, se])
    # merge_part = tf.keras.layers.GlobalAveragePooling1D()(x)

    pi_part = build_conv_block(shape=(21, embed_dim))(embedding(inputs_pi))
    m_part = build_conv_block(shape=(31, embed_dim))(embedding(inputs_m))

    merge = tf.concat([pi_part, m_part], axis=1)
    # merge = tf.add(pi_part, m_part)
    x = tf.keras.layers.BatchNormalization()(merge)
    x = tf.keras.layers.Dense(128, activation='relu', kernel_initializer='he_normal')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(64, activation='relu', kernel_initializer='he_normal')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(32, activation='relu', kernel_initializer='he_normal')(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(inputs=[inputs_pi, inputs_m], outputs=x)
    model.summary()
    return model
# }}}

