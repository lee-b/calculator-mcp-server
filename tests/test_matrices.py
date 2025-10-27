import pytest
import numpy as np
from calculator_mcp_server import (
    matrix_addition,
    matrix_multiplication,
    matrix_transpose,
    matrix_determinant,
)


class TestMatrixAddition:
    """Test cases for the matrix_addition function."""

    def test_basic_addition(self, sample_matrix_2x2):
        matrix_b = [[5, 6], [7, 8]]
        result = matrix_addition(sample_matrix_2x2, matrix_b)
        expected = np.add(sample_matrix_2x2, matrix_b).tolist()
        assert result == {"result": expected}

    def test_addition_with_zero_matrix(self, sample_matrix_2x2, zero_matrix):
        result = matrix_addition(sample_matrix_2x2, zero_matrix)
        assert result == {"result": sample_matrix_2x2}

    def test_addition_identity(self, sample_matrix_2x2, identity_matrix):
        result = matrix_addition(sample_matrix_2x2, identity_matrix)
        expected = [[2, 2], [3, 5]]
        assert result == {"result": expected}

    def test_different_sizes_error(self, sample_matrix_2x2, sample_matrix_2x3):
        result = matrix_addition(sample_matrix_2x2, sample_matrix_2x3)
        assert "error" in result

    def test_empty_matrix_error(self):
        result = matrix_addition([], [])
        assert "error" in result


class TestMatrixMultiplication:
    """Test cases for the matrix_multiplication function."""

    def test_basic_multiplication(self, sample_matrix_2x2):
        matrix_b = [[5, 6], [7, 8]]
        result = matrix_multiplication(sample_matrix_2x2, matrix_b)
        expected = np.dot(sample_matrix_2x2, matrix_b).tolist()
        assert result == {"result": expected}

    def test_multiplication_by_identity(self, sample_matrix_2x2, identity_matrix):
        result = matrix_multiplication(sample_matrix_2x2, identity_matrix)
        assert result == {"result": sample_matrix_2x2}

    def test_incompatible_dimensions_error(self, sample_matrix_2x2, sample_matrix_2x3):
        # 2x2 * 2x3 should work
        result = matrix_multiplication(sample_matrix_2x2, sample_matrix_2x3)
        expected = np.dot(sample_matrix_2x2, sample_matrix_2x3).tolist()
        assert result == {"result": expected}

    def test_wrong_dimensions_error(self, sample_matrix_2x3, sample_matrix_2x2):
        # 2x3 * 2x2 should fail
        result = matrix_multiplication(sample_matrix_2x3, sample_matrix_2x2)
        assert "error" in result

    def test_empty_matrix_error(self):
        result = matrix_multiplication([], [])
        assert "error" in result


class TestMatrixTranspose:
    """Test cases for the matrix_transpose function."""

    def test_basic_transpose(self, sample_matrix_2x2):
        result = matrix_transpose(sample_matrix_2x2)
        expected = np.transpose(sample_matrix_2x2).tolist()
        assert result == {"result": expected}

    def test_transpose_rectangle(self, sample_matrix_2x3):
        result = matrix_transpose(sample_matrix_2x3)
        expected = np.transpose(sample_matrix_2x3).tolist()
        assert result == {"result": expected}

    def test_transpose_identity(self, identity_matrix):
        result = matrix_transpose(identity_matrix)
        assert result == {"result": identity_matrix}

    def test_transpose_empty_error(self):
        result = matrix_transpose([])
        assert "error" in result

    def test_double_transpose(self, sample_matrix_2x2):
        result = matrix_transpose(matrix_transpose(sample_matrix_2x2)["result"])
        assert result == {"result": sample_matrix_2x2}


class TestMatrixDeterminant:
    """Test cases for the matrix_determinant function."""

    def test_basic_determinant(self, sample_matrix_2x2):
        result = matrix_determinant(sample_matrix_2x2)
        expected = round(float(np.linalg.det(sample_matrix_2x2)), 10)
        assert result == {"result": expected}

    def test_identity_determinant(self, identity_matrix):
        result = matrix_determinant(identity_matrix)
        assert result == {"result": 1.0}

    def test_zero_determinant(self, zero_matrix):
        result = matrix_determinant(zero_matrix)
        assert result == {"result": 0.0}

    def test_3x3_determinant(self, matrix_3x3):
        result = matrix_determinant(matrix_3x3)
        expected = round(float(np.linalg.det(matrix_3x3)), 10)
        assert result == {"result": expected}

    def test_non_square_error(self, sample_matrix_2x3):
        result = matrix_determinant(sample_matrix_2x3)
        assert "error" in result

    def test_empty_matrix_error(self):
        result = matrix_determinant([])
        assert "error" in result

    def test_single_element(self):
        matrix = [[5]]
        result = matrix_determinant(matrix)
        assert result == {"result": 5.0}