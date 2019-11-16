from acouSceneClassification import *
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import os
#import mnist
import sys

def listdir_nohidden(path, numFiles):
    for f in os.listdir(path)[:numFiles]:
        if not f.startswith('.'):
            yield f


def indices_to_one_hot(data, nb_classes):
    """Convert an iterable of indices to one-hot encoded labels."""
    targets = np.array(data).reshape(-1)
    return np.eye(nb_classes)[targets]


def load_data_mnist():
    train_images = mnist.train_images()
    train_labels = mnist.train_labels()
    test_images = mnist.test_images()
    test_labels = mnist.test_labels()

    train_images = train_images / 255.0
    test_images = test_images / 255.0
    train_images = train_images[:, :, :, np.newaxis]
    test_images = test_images[:, :, :, np.newaxis]
    train_labels = indices_to_one_hot(train_labels, 10)
    test_labels = indices_to_one_hot(test_labels, 10)

    return train_images, test_images, train_labels, test_labels


def load_data(mypath):
    X = []
    y = []
    numClass = 3
    input_size = 441000
    numFiles = 600
    scaler = MinMaxScaler(feature_range=(-1, 1))

    for idx in range(0, numClass):
        currDirectory = mypath + '-' + str(idx) + '/audio'
        listFiles = listdir_nohidden(currDirectory, numFiles)
        for file in listFiles:
            currFile = currDirectory + '/' + file
            src, sr = librosa.load(currFile, sr=None, mono=True)
            X.append(src[:input_size])
            y.append([idx])

    # Shape data
    X = np.array(X)
    y_ = indices_to_one_hot(y, numClass)

    # Split into training and testing set
    X_train, X_test, y_train, y_test = train_test_split(X, y_, test_size=0.2)

    # Normalize data
    X_train = np.array(X_train)
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    X_train = X_train[:,np.newaxis,:]
    X_test = X_test[:, np.newaxis, :]

    return X_train, X_test, y_train, y_test


def main():
    model_path = sys.argv[1]
    data_path = sys.argv[2]

    # Load data
    X_train, X_test, y_train, y_test = load_data(data_path)
    #X_train, X_test, y_train, y_test = load_data_mnist()

    # Pre-process data
    input_shape = X_train[0].shape
    m = acouSceneClassification(input_shape, epochs=20, batch_size=16)
    m.preprocess_data(X_train)

    # Build model
    m.model_build()

    # Train model
    m.model_training(y_train)

    # Testing model
    m.model_evaluate(X_test, y_test)

    m.model_save(model_path, '3_classes_model.h5')


if __name__ == '__main__':
    main()