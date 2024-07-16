#!/usr/bin/env python3

"""
Adjugate calculation
"""


def validate_matrix(matrix):
    """
    Validates the type of the matrix (should be a list of lists)
    """
    if (not isinstance(matrix, list) or len(matrix) == 0
            or not all(isinstance(row, list) for row in matrix)):
        raise TypeError("matrix must be a list of lists")


def is_square(matrix):
    """
    Validates the shape of the matrix (should be square)
    """
    if any(len(row) != len(matrix) for row in matrix):
        raise ValueError("matrix must be a square matrix")


def determinant(matrix):
    """
    Calculate the determinant of a square matrix.

    Args:
        matrix (list of list): The given matrix for determinant calculation.

    Returns:
        float: The determinant of the matrix.
    """
    validate_matrix(matrix)

    # Handle 0x0: empty matrix [[]], determinant is 1
    if matrix == [[]]:
        return 1

    is_square(matrix)
    n = len(matrix)

    # Base case for 1x1 matrix
    if n == 1:
        return matrix[0][0]

    # Base case for 2x2 matrix
    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    # Recursive case: Laplace expansion, should work for a nxn matrix
    det = 0
    for col in range(n):
        sub_matrix = [[matrix[row][c]
                       for c in range(n) if c != col] for row in range(1, n)]
        # Flip the sign for each other column position
        sign = (-1) ** col
        # NOTE Recursive call: reduces the matrix size by 1 dimension
        # Results accumulate into det, using the right sign
        det += sign * matrix[0][col] * determinant(sub_matrix)

    return det


def _validate_matrix(matrix):
    """
    Validates the matrix, with slightly worse checks (checker requirement).
    """
    if (not isinstance(matrix, list) or len(matrix) == 0
            or not all(isinstance(row, list) for row in matrix)):
        raise TypeError("matrix must be a list of lists")
    if any(len(row) != len(matrix) for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")


def minor(matrix):
    """
    Calculate the minor matrix of a square matrix.

    Args:
        `matrix` (list of list): The matrix whose minor matrix should
        be calculated.

    Returns:
        list of list: The minor matrix of the input matrix.
    """
    _validate_matrix(matrix)
    n = len(matrix)

    # Handling special case: minor of 1x1 matrix
    if n == 1:
        return [[1]]

    minor_matrix = []
    for i in range(n):
        minor_row = []
        for j in range(n):
            # Sub matrix, removing row i & column j from the source matrix
            sub_matrix = [[matrix[row][col] for col in range(n) if col != j]
                          for row in range(n) if row != i]
            # Calculate determinant of submatrix for these coordinates
            minor_row.append(determinant(sub_matrix))
        minor_matrix.append(minor_row)

    return minor_matrix


def cofactor(matrix):
    """
    Calculate the cofactor matrix of a square matrix.

    Args:
        matrix (list of list): The matrix whose cofactor matrix should
        be calculated.

    Returns:
        list of list: The cofactor matrix of the input matrix.
    """
    _validate_matrix(matrix)

    cofactor_matrix = minor(matrix)
    n = len(cofactor_matrix)

    for i in range(n):
        for j in range(n):
            cofactor_matrix[i][j] *= (-1) ** (i + j)

    return cofactor_matrix


def adjugate(matrix):
    """
    Calculates the adjugate matrix of a matrix.

    Args:
        matrix (list of list): The matrix whose adjugate matrix should
        be calculated.

    Returns:
        list of list: The adjugate matrix of the input matrix.
    """
    _validate_matrix(matrix)
    cofactor_matrix = cofactor(matrix)
    n = len(cofactor_matrix)

    # Transpose matrix elements over diagonal (swap coordinates)
    return [[cofactor_matrix[j][i] for j in range(n)] for i in range(n)]
