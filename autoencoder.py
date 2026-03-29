from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2


class autoencoder:

    def __init__(self, input_dim, verbose=True):

        input_layer = Input(shape=(input_dim,))

        # 🔥 ENCODER
        x = Dense(32, activation='relu', kernel_regularizer=l2(0.0001))(input_layer)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)

        x = Dense(16, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)

        x = Dense(6, activation='relu')(x)

        # 🔥 DECODER
        x = Dense(16, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)

        x = Dense(32, activation='relu')(x)

        output = Dense(input_dim, activation='sigmoid')(x)

        self.model = Model(inputs=input_layer, outputs=output)

        self.model.compile(
            optimizer='adam',
            loss='mse'
        )

        if verbose:
            self.model.summary()