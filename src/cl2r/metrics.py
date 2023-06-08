import numpy as np 


def average_compatibility(matrix=None):
    steps = matrix.shape[0]
    position = np.zeros_like(matrix, dtype=bool)
    for j in range(matrix.shape[0]):
        for i in range(j + 1, matrix.shape[1]):
            if matrix[i][j] < matrix[j][j]:
                position[i, j] = True
    max_ac = (steps * (steps-1)) / 2
    if max_ac < 1:
        max_ac = 1
    ac = max_ac - np.sum(position)
    return (1/max_ac) * ac


def backward_compatibility(matrix):
    bc = 0
    for i in range(matrix.shape[0]-1):
        bc+=matrix[-1][i]-matrix[i][i]
    return bc/(matrix.shape[0]-1)


def forward_compatibility(matrix):
    fc = 0
    for i in range(1,matrix.shape[0]):
        fc+= matrix[i][i-1] - matrix[i][i]
    return fc/(matrix.shape[0]-1)
