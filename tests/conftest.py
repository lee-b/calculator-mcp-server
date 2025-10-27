import pytest
import numpy as np
import warnings

# Suppress specific warnings in tests
warnings.filterwarnings("ignore", message="invalid value encountered in divide", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="One or more sample arguments is too small", category=UserWarning)


@pytest.fixture
def sample_data():
    """Fixture providing sample numerical data for statistical tests."""
    return [1.0, 2.0, 3.0, 4.0, 5.0]


@pytest.fixture
def sample_matrix_2x2():
    """Fixture providing a 2x2 matrix."""
    return [[1, 2], [3, 4]]


@pytest.fixture
def sample_matrix_2x3():
    """Fixture providing a 2x3 matrix."""
    return [[1, 2, 3], [4, 5, 6]]


@pytest.fixture
def sample_vector_3d():
    """Fixture providing a 3D vector."""
    return (1, 2, 3)


@pytest.fixture
def sample_points():
    """Fixture providing sample points for linear regression."""
    return [(1, 2), (2, 3), (3, 5), (4, 7)]


@pytest.fixture
def empty_data():
    """Fixture providing empty data for edge case tests."""
    return []


@pytest.fixture
def single_element_data():
    """Fixture providing single element data."""
    return [5.0]


@pytest.fixture
def large_data():
    """Fixture providing large dataset."""
    return list(np.random.rand(1000))


@pytest.fixture
def matrix_3x3():
    """Fixture providing a 3x3 matrix."""
    return [[1, 2, 3], [4, 5, 6], [7, 8, 9]]


@pytest.fixture
def identity_matrix():
    """Fixture providing identity matrix."""
    return [[1, 0], [0, 1]]


@pytest.fixture
def zero_matrix():
    """Fixture providing zero matrix."""
    return [[0, 0], [0, 0]]