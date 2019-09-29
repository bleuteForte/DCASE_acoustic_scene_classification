from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import MaxPooling2D, Dropout
from keras.models import Model
from kapre.time_frequency import Spectrogram
import numpy as np


class acouSceneClassification:

    def __init__(self, input_shape, epochs=20, batch_size=32):
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = self.Model_build(input_shape=input_shape)

    def Model_build(self, input_shape):
        """
        Implementation of the model.

        Arguments:
        input_shape -- shape of the images of the dataset

        Returns:
        model -- a Model() instance in Keras
        """

        # Define the input placeholder as a tensor with shape input_shape. Think of this as your input image!
        X_input = Input(input_shape)

        X = Spectrogram(n_dft=512, n_hop=256, input_shape=input_shape,
                        return_decibel_spectrogram=True, power_spectrogram=2.0,
                        trainable_kernel=False, name='static_stft')(X_input)

        # CONV -> BN -> RELU Block applied to X
        X = Conv2D(8, (8, 8), name='conv0')(X)
        X = BatchNormalization(name='bn0')(X)
        X = Activation('relu')(X)
        X = MaxPooling2D((2, 4), name='max_pool0')(X)
        X = Dropout(0.1, name='dropout0')(X)

        X = Conv2D(16, (16, 16), name='conv1')(X)
        X = BatchNormalization(name='bn1')(X)
        X = Activation('relu')(X)
        X = MaxPooling2D((2, 4), name='max_pool1')(X)
        X = Dropout(0.1, name='dropout1')(X)

        X = Conv2D(16, (32, 32), name='conv2')(X)
        X = BatchNormalization(name='bn2')(X)
        X = Activation('relu')(X)
        X = MaxPooling2D((2, 4), name='max_pool2')(X)
        X = Dropout(0.1, name='dropout2')(X)

        # FLATTEN X (means convert it to a vector) + FULLYCONNECTED
        X = Flatten()(X)
        X = Dense(32, activation='relu', name='fc0')(X)
        X = Dense(10, activation='softmax', name='fc1')(X)

        # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
        model = Model(inputs=X_input, outputs=X, name='acouModel')

        return model

    def Model_training(self, X, y):
        self.model.compile(optimizer='adam', loss='mean_squared_error', metrics=["accuracy"])
        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size)

