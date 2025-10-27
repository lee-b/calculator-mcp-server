import pytest
import numpy as np
from calculator_mcp_server import (
    vector_dot_product,
    vector_cross_product,
    vector_magnitude,
)


class TestVectorDotProduct:
    """Test cases for the vector_dot_product function."""

    def test_basic_dot_product(self, sample_vector_3d):
        vector_b = (4, 5, 6)
        result = vector_dot_product(sample_vector_3d, vector_b)
        expected = np.dot(sample_vector_3d, vector_b)
        assert result == {"result": expected}

    def test_orthogonal_vectors(self):
        a = (1, 0, 0)
        b = (0, 1, 0)
        result = vector_dot_product(a, b)
        assert result == {"result": 0}

    def test_same_vector(self, sample_vector_3d):
        result = vector_dot_product(sample_vector_3d, sample_vector_3d)
        expected = np.dot(sample_vector_3d, sample_vector_3d)
        assert result == {"result": expected}

    def test_zero_vector(self, sample_vector_3d):
        zero_vec = (0, 0, 0)
        result = vector_dot_product(sample_vector_3d, zero_vec)
        assert result == {"result": 0}

    def test_2d_vectors(self):
        a = (1, 2)
        b = (3, 4)
        result = vector_dot_product(a, b)
        assert result == {"result": 11}  # 1*3 + 2*4

    def test_different_dimensions_error(self, sample_vector_3d):
        vec_2d = (1, 2)
        result = vector_dot_product(sample_vector_3d, vec_2d)
        assert "error" in result

    def test_empty_vectors_error(self):
        result = vector_dot_product((), ())
        assert "error" in result


class TestVectorCrossProduct:
    """Test cases for the vector_cross_product function."""

    def test_basic_cross_product(self, sample_vector_3d):
        vector_b = (4, 5, 6)
        result = vector_cross_product(sample_vector_3d, vector_b)
        expected = np.cross(sample_vector_3d, vector_b).tolist()
        assert result == {"result": expected}

    def test_standard_basis(self):
        i = (1, 0, 0)
        j = (0, 1, 0)
        result = vector_cross_product(i, j)
        assert result == {"result": [0, 0, 1]}

    def test_parallel_vectors(self, sample_vector_3d):
        # Cross product of parallel vectors is zero
        result = vector_cross_product(sample_vector_3d, sample_vector_3d)
        assert result == {"result": [0, 0, 0]}

    def test_2d_vectors_error(self):
        a = (1, 2)
        b = (3, 4)
        result = vector_cross_product(a, b)
        assert "error" in result  # Cross product only defined for 3D

    def test_different_dimensions_error(self, sample_vector_3d):
        vec_2d = (1, 2)
        result = vector_cross_product(sample_vector_3d, vec_2d)
        assert "error" in result

    def test_empty_vectors_error(self):
        result = vector_cross_product((), ())
        assert "error" in result


class TestVectorMagnitude:
    """Test cases for the vector_magnitude function."""

    def test_basic_magnitude(self, sample_vector_3d):
        result = vector_magnitude(sample_vector_3d)
        expected = np.linalg.norm(sample_vector_3d)
        assert result == {"result": expected}

    def test_unit_vector_x(self):
        vec = (1, 0, 0)
        result = vector_magnitude(vec)
        assert result == {"result": 1.0}

    def test_zero_vector(self):
        vec = (0, 0, 0)
        result = vector_magnitude(vec)
        assert result == {"result": 0.0}

    def test_2d_vector(self):
        vec = (3, 4)
        result = vector_magnitude(vec)
        assert result == {"result": 5.0}

    def test_negative_components(self):
        vec = (-3, -4)
        result = vector_magnitude(vec)
        assert result == {"result": 5.0}

    def test_large_vector(self):
        vec = (1, 2, 3, 4, 5)
        result = vector_magnitude(vec)
        expected = np.linalg.norm(vec)
        assert result == {"result": expected}

    def test_empty_vector_error(self):
        result = vector_magnitude(())
        assert "error" in result