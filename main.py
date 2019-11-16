from acouSceneClassification import *
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import os
import mnist

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
    numFiles = 60
    scaler = MinMaxScaler(feature_range=(-1, 1))

    for idx in range(0, numClass):
        currDirectory = mypath + '-' + str(idx) + '/audio'
        listFiles = listdir_nohidden(currDirectory, numFiles)
        for file in listFiles:
            currFile = currDirectory + '/' + file
            src, sr = librosa.load(currFile, sr=None, mono=True)
            X.append(src[:input_size])
            y.append([idx])

    # Normalize data
    X = np.array(X)
    scaler.fit(X)
    X = scaler.transform(X)

    # Shape data
    X = X[:, np.newaxis, :]
    y_ = indices_to_one_hot(y, numClass)

    # Split into training and testing set
    X_train, X_test, y_train, y_test = train_test_split(X, y_, test_size=0.2)
    return X_train, X_test, y_train, y_test


def training_model(X_train, y_train):
    input_shape = X_train[0].shape
    m = acouSceneClassification(input_shape, epochs=1, batch_size=2)
    m.Model_training(X_train, y_train)
    return m

def testing_model(m, X_test, y_test):
    preds = m.model.evaluate(x=X_test, y=y_test, batch_size=32)

    print()
    print("Loss = " + str(preds[0]))
    print("Test Accuracy = " + str(preds[1]))

def main():

    # Load data
    mypath = "/Users/bassed/Downloads/TAU-urban-acoustic-scenes-2019-evaluation"
    X_train, X_test, y_train, y_test = load_data(mypath)
    #X_train, X_test, y_train, y_test = load_data_mnist()

    # Pre-process data
    input_shape = X_train[0].shape
    m = acouSceneClassification(input_shape, epochs=1, batch_size=16)
    m.preprocess_data(X_train)

    # Build model
    m.Model_build()

    # Train model
    m.Model_training(y_train)

    # Testing model
    #m.Model_testing()

    #testing_model(m, X_test, y_test)

if __name__ == '__main__':
    main()