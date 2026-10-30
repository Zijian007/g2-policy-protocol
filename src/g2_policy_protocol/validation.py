"""Rigid calibration validation without model/runtime dependencies."""

import math


def validate_rigid_transform(matrix, name):
    if (
        not isinstance(matrix, list)
        or len(matrix) != 4
        or any(not isinstance(row, list) or len(row) != 4 for row in matrix)
    ):
        raise ValueError(f"{name} must be a 4x4 rigid transform")
    if any(type(v) not in (int, float) or not math.isfinite(v) for row in matrix for v in row):
        raise ValueError(f"{name} must contain finite numbers")
    if any(abs(a - b) > 1e-6 for a, b in zip(matrix[3], [0, 0, 0, 1], strict=True)):
        raise ValueError(f"{name} must have homogeneous last row [0,0,0,1]")
    for i in range(3):
        for j in range(3):
            if abs(sum(matrix[k][i] * matrix[k][j] for k in range(3)) - int(i == j)) > 1e-5:
                raise ValueError(f"{name} rotation must be orthonormal")
    r = matrix
    determinant = (
        r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1])
        - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0])
        + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0])
    )
    if abs(determinant - 1) > 1e-5:
        raise ValueError(f"{name} rotation determinant must be +1")
