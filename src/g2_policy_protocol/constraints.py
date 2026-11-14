"""Foundational geometric constraints shared by wire and training contracts.

Standard library only. This module validates geometry without knowing about
policy configuration, dataset columns or transport implementations.
"""

import math


def validate_rigid_transform(matrix: list[list[float]], name: str) -> None:
    """Require a finite 4x4 transform with a proper orthonormal rotation.

    The final row is [0,0,0,1]; reflections, scales and shears are rejected.
    ``name`` identifies the input field in any ValueError.
    """
    if (
        not isinstance(matrix, list)
        or len(matrix) != 4
        or any(not isinstance(row, list) or len(row) != 4 for row in matrix)
    ):
        raise ValueError(f"{name} must be a 4x4 rigid transform")
    if any(type(component) not in (int, float) or not math.isfinite(component) for row in matrix for component in row):
        raise ValueError(f"{name} must contain finite numbers")
    if any(abs(actual - expected) > 1e-6 for actual, expected in zip(matrix[3], [0, 0, 0, 1], strict=True)):
        raise ValueError(f"{name} must have homogeneous last row [0,0,0,1]")
    for column in range(3):
        for other_column in range(3):
            if (
                abs(
                    sum(matrix[row][column] * matrix[row][other_column] for row in range(3))
                    - int(column == other_column)
                )
                > 1e-5
            ):
                raise ValueError(f"{name} rotation must be orthonormal")
    determinant = (
        matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
    )
    if abs(determinant - 1) > 1e-5:
        raise ValueError(f"{name} rotation determinant must be +1")
