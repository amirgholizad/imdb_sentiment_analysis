# import packages
import tensorflow as tf
from keras.preprocessing.text import Tokenizer


# One-hot encoding the output into vector mode, each of length 100
def imdb_preprocessing(
    features_train, features_test, targets_train, targets_test, n_words, n_classes
):
    tokenizer = Tokenizer(num_words=n_words)
    x_train = tokenizer.sequences_to_matrix(features_train, mode="binary")
    x_test = tokenizer.sequences_to_matrix(features_test, mode="binary")

    # One-hot encoding the output
    y_train = tf.keras.utils.to_categorical(targets_train, n_classes)
    y_test = tf.keras.utils.to_categorical(targets_test, n_classes)

    return x_train, x_test, y_train, y_test
