import ast


def load_n_grams(file_path):
    """
    Load n grams from text files
    :param file_path: input to bi-gram or tri-gram file
    :return: n-gram words. E.g. bi-gram words or tri-gram words
    """
    with open(file_path, encoding="utf8") as fr:
        words = fr.read()
        words = ast.literal_eval(words)
    return words
