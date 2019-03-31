import string
import random
import torch


# List of strings considered for this experiment
all_strings = string.ascii_letters + " .,;'\"\n\t"
n_letters = len(all_strings)
all_categories = [1, 0]


def random_string():
    """
    Generates random strings
    :return: str
    """
    random_length = random.randint(0, n_letters)
    random_char_list = random.sample(list(all_strings), random_length)
    return ''.join(random_char_list)


def letter_to_index(letter):
    """
    Return the position of letter if found in all_strings

    :param letter: str
    :return: int
    """
    return all_strings.find(letter)


def letter_to_tensor(letter):
    """
    Returns a 1-hot encoded vector as a torch tensor

    :param letter: str
    :return: torch.tensor
    """
    tensor = torch.zeros(1, n_letters)
    tensor[0][letter_to_index(letter)] = 1
    return tensor


def line_to_tensor(line):
    """
    Returns matrix representing 1-hot encoded vectors for characters in the string

    :param line: str
    :return: torch.tensor
    """
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letter_to_index(letter)] = 1
    return tensor


def category_from_output(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i
