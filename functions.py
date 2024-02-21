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


# preprocessing input and making predictions
def imdb_predict(model, dataset, review, n_words):
    tokenizer = Tokenizer(num_words=n_words)
    word_index = dataset.get_word_index()

    review_list = review.lower().split()
    temp = []
    for word in review_list:
        temp.append("".join(c for c in word if c.isalpha()))
    review_list = temp

    encoded_review = [word_index[word] for word in review_list]
    encoded_review = tokenizer.sequences_to_matrix(
        [encoded_review], mode="binary"
    ).squeeze()
    encoded_review = tf.keras.preprocessing.sequence.pad_sequences(
        [encoded_review], maxlen=n_words
    )

    pred = model.predict(encoded_review)
    if pred[0][1] > 0.5:
        print("This review was Positive!")
    elif pred[0][0] > 0.5:
        print("This review was Negative!")
