from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import MaxPooling2D, Dropout
from keras.models import Model, load_model
#import essentia
#from essentia.standard import *
from librosa.filters import get_window
from librosa.feature import melspectrogram
#from kapre.time_frequency import Spectrogram
import numpy as np
from keras.callbacks import EarlyStopping


class acouSceneClassification:

    def __init__(self, input_shape, epochs=20, batch_size=32):
        self.epochs = epochs
        self.batch_size = batch_size
        self.input_shape = input_shape
        #self.mod   el = self.Model_build(input_shape=input_shape)
        self.Nx = 512
        self.w = get_window(window='hann', Nx=self.Nx)
        #self.spectrum = Spectrum()  # FFT() would return the complex FFT, here we just want the magnitude spectrum
        #self.mfcc = MFCC()
        self.numPreprocessFrames = 10
        self.frameSize = input_shape[1] / self.numPreprocessFrames
        self.inputData = []
        self.model = 0

    def preprocess_data(self, x):
        melbands = []
        for signal in x:
            signal = np.squeeze(signal)
            mel_spec = melspectrogram(y=signal, n_fft=44100, hop_length=44100, n_mels=40)
            melbands.append(np.reshape(mel_spec,(mel_spec.shape[0]*mel_spec.shape[1],),order='F'))
        self.inputData = np.array(melbands)

    def model_build(self):
        """
        Implementation of the model.

        Returns:
        model -- a Model() instance in Keras
        """

        # Define the input placeholder as a tensor with shape input_shape. Think of this as your input image!
        X_input = Input(self.inputData[0].shape)

        '''
        # CONV -> BN -> RELU Block applied to X
        X = Conv2D(8, (8, 8), name='conv0')(X_input)
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
'       '''

        X = Dense(500, activation='relu', name='fc0')(X_input)
        X = Dropout(0.1, name='dropout1')(X)
        X = Dense(500, activation='relu', name='fc1')(X)
        X = Dropout(0.1, name='dropout2')(X)
        X = Dense(3, activation='softmax', name='fc2')(X)

        # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
        self.model = Model(inputs=X_input, outputs=X, name='acouModel')

    def model_training(self, y):
        callbacks = [EarlyStopping(monitor='val_loss', patience=2)]
        self.model.compile(optimizer='RMSProp', loss='categorical_crossentropy', metrics=["accuracy"])
        self.model.fit(self.inputData, y, epochs=self.epochs, batch_size=self.batch_size, callbacks=callbacks,
                       validation_split=0.1)

    def model_evaluate(self, X_test, y_test):
        self.preprocess_data(X_test)
        preds = self.model.evaluate(x=self.inputData, y=y_test)

        print()
        print("Loss = " + str(preds[0]))
        print("Test Accuracy = " + str(preds[1]))

    def model_save(self, path, filename):
        self.model.save(path+filename)

    def model_load(self, path):
        self.model = load_model(path)