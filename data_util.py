from collections import Counter
import nltk
from nltk import SnowballStemmer
import pandas as pd
from sklearn import metrics


def get_prfa(dev_y: list, pred_y: list, verbose=False) -> tuple:
    """
    Calculate precision, recall, f1, and accuracy for a given set of predictions and labels.
    Args:
        dev_y: list of labels
        pred_y: list of predictions
        verbose: whether to print the metrics
    Returns:
        tuple of precision, recall, f1, and accuracy
    """
    precision = metrics.precision_score(dev_y, pred_y)
    recall = metrics.recall_score(dev_y, pred_y)
    f1 = metrics.f1_score(dev_y, pred_y)
    accuracy = metrics.accuracy_score(dev_y, pred_y)
    if verbose:
        print(f'Precision: {precision}')
        print(f'Recall: {recall}')
        print(f'F1 Score: {f1}')
        print(f'Accuracy: {accuracy}')
    return precision, recall, f1, accuracy


def probs_to_preds(probabilities: list) -> list[int]:
    """
    Converts continuous (sigmoid) outputs to discrete binary probabilities
    :param probabilities: List of probabilities between [0, 1]
    :return:
    """
    return [1 if p[0] > 0.5 else 0 for p in probabilities]


def generate_tuples_from_df(df: pd.DataFrame) -> tuple[list[list[str]], list[int]]:
    """
    Generates data from Pandas DataFrame in format:

    tokenized text from file: [[word1, word2, ...], [word1, word2, ...], ...]
    labels: [0, 1, 0, 1, ...]

    :param df: The Pandas DataFrame
    :return: A list of lists of tokens, list of integer labels
    """
    X = []
    y = []
    for text, label in df.itertuples(index=False):
        if len(text.strip()) == 0:
            continue
        else:
            X.append(nltk.word_tokenize(text))
            y.append(int(label))
    return X, y


def generate_tuples_from_file(training_file_path: str) -> tuple[list[list[str]], list[int]]:
    """
    Generates data from file formatted like:

    tokenized text from file: [[word1, word2, ...], [word1, word2, ...], ...]
    labels: [0, 1, 0, 1, ...]

    Parameters:
        training_file_path - Path to the training file(s)
    Return:
        A list of lists of tokens and a list of int labels
    """
    if training_file_path.endswith(".csv"):
        X = []
        y = []
        training_df = pd.read_csv(training_file_path)
        for text, label in training_df.itertuples(index=False):
            if len(text.strip()) == 0:
                continue
            else:
                X.append(nltk.word_tokenize(text))
                y.append(int(label))
        return X, y
    else:
        training_file = open(training_file_path, "r", encoding="utf8")
        X = []
        y = []
        for sentence in training_file:
            if len(sentence.strip()) == 0:
                continue
            data_in_sentence = sentence.strip().split("\t")
            if len(data_in_sentence) != 3:
                continue
            else:
                t = tuple(data_in_sentence)
                if (not t[2] == '0') and (not t[2] == '1'):
                    print("WARNING")
                    continue
                X.append(nltk.word_tokenize(t[1]))
                y.append(int(t[2]))
        training_file.close()
        return X, y


def create_vocabulary(training_data_X: list) -> list:
    """
    Given the training data, create a list of all the words in the training data.
    Args:
        training_data_X: a list of all the training data in the format [[word1, word2, ...], ...]
    Returns:
        vocab: a list of all the unique words in the training data
    """
    vocabulary = set()
    stemmer = SnowballStemmer("english")
    for document in training_data_X:
        for word in document:
            vocabulary.add(stemmer.stem(word))
    return list(vocabulary)


def featurize(vocab: list, data_to_be_featurized_X: list, binary: bool = False, verbose: bool = False) -> list:
    """
    Create vectorized BoW representations of the given data.
    Args:
        vocab: list of words in vocabulary
        data_to_be_featurized_X: a list of data to be featurized in the format [[word1, word2, ...], ...]
        binary: whether to use binary features
        verbose: boolean for whether to print out progress
    Returns:
        a list of sparse vector representations of the data in the format [[count1, count2, ...], ...]
    """
    X = []
    for document in data_to_be_featurized_X:
        word_counts = Counter(document)
        document_features = {}
        for word in vocab:
            if binary:
                # Convert true or false to integer
                document_features[word] = int(word in word_counts)
            else:
                # If not binary, get counts directly and default to 0
                document_features[word] = word_counts.get(word, 0)
            if verbose:
                print(f'Add {word} with {document_features[word]}')
        X.append(document_features)
    return X
