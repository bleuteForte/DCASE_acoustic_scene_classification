from acouSceneClassification import *
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def indices_to_one_hot(data, nb_classes):
    """Convert an iterable of indices to one-hot encoded labels."""
    targets = np.array(data).reshape(-1)
    return np.eye(nb_classes)[targets]


def load_data():
    numTrainData = 20
    input_size = 441000
    # Import audio files
    X = []
    y = []
    for idx in range(0, numTrainData):
        currentFile = 'data/training_set/' + str(idx) + '.wav'
        src, sr = librosa.load(currentFile, sr=None, mono=True)

        '''
        # Make it 2D
        len_second = src.shape[0] / sr
        src = src[:int(sr*len_second)]
        src = src[np.newaxis, :]
        '''

        X.append(src[:input_size])
        y.append([np.random.randint(10)])

    X = np.array(X)
    X = X[:, np.newaxis, :]

    y_ = indices_to_one_hot(y, 10)

    X_train, X_test, y_train, y_test = train_test_split(X, y_, test_size=0.2)
    return X_train, X_test, y_train, y_test


def training_model(X_train, X_test, y_train, y_test):
    input_shape = X_train[0].shape
    m = acouSceneClassification(input_shape, epochs=1, batch_size=2)
    m.Model_training(X_train, y_train)

    m.model.predict(X_test)


def main():
    X_train, X_test, y_train, y_test = load_data()
    training_model(X_train, X_test, y_train, y_test)


if __name__ == '__main__':
    main()