import numpy as np
import json


def load_json(filename):
    with open(filename, 'r') as opened_file:
        return json.load(opened_file)


def logical_and(array_list):
    array = array_list[0]
    for other_array in array_list[1:]:
        array = np.logical_and(array, other_array)
    return array


def logical_or(array_list):
    array = array_list[0]
    for other_array in array_list[1:]:
        array = np.logical_or(array, other_array)
    return array


def get_index_value(iterable, index):
    if iterable:
        return iterable[index]
