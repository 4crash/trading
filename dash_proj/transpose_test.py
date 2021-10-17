from typing import List
import numpy as np


def transpose(matrix: np):
    return matrix.transpose()


matrix = np.array([['a','b','c'],['e','f']], dtype=str)
output = transpose(matrix)
print(output)