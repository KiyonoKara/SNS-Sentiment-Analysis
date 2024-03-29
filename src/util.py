from collections import Counter


def create_index(all_train_data_X: list) -> list:
    """
    Given the training data, create a list of all the words in the training data.
    Args:
        all_train_data_X: a list of all the training data in the format [[word1, word2, ...], ...]
    Returns:
        vocab: a list of all the unique words in the training data
    """
    vocabulary = {word for document in all_train_data_X for word in document}
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
