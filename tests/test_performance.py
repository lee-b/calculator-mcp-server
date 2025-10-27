import pytest
import time
import concurrent.futures
import numpy as np
from calculator_mcp_server import (
    calculate,
    matrix_multiplication,
    matrix_addition,
    matrix_transpose,
    matrix_determinant,
    vector_dot_product,
    vector_magnitude,
    summation,
)


class TestConcurrentToolCalls:
    """Test cases for concurrent execution of tool calls."""

    def test_concurrent_calculations(self):
        """Test running multiple calculate calls concurrently."""
        expressions = ["2 + 3", "4 * 5", "10 / 2", "2 ** 3", "sin(pi/2)", "sqrt(16)"] * 10  # 60 calls

        start_time = time.time()

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(calculate, expr) for expr in expressions]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]

        end_time = time.time()
        duration = end_time - start_time

        # Assert all results are successful
        assert all("result" in r for r in results)
        # Assert reasonable performance (should complete in under 5 seconds)
        assert duration < 5.0

    def test_concurrent_matrix_operations(self):
        """Test running multiple matrix operations concurrently."""
        # Create sample matrices
        matrices = []
        for i in range(10):
            matrices.append([[i+1, i+2], [i+3, i+4]])

        start_time = time.time()

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(matrix_multiplication, m, m) for m in matrices]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]

        end_time = time.time()
        duration = end_time - start_time

        # Assert all results are successful
        assert all("result" in r for r in results)
        # Assert reasonable performance
        assert duration < 5.0


class TestLargeDataHandling:
    """Test cases for handling large data inputs."""

    def test_large_matrix_multiplication(self):
        """Test matrix multiplication with large matrices (100x100)."""
        size = 100
        # Create large matrices
        matrix_a = np.random.rand(size, size).tolist()
        matrix_b = np.random.rand(size, size).tolist()

        start_time = time.time()
        result = matrix_multiplication(matrix_a, matrix_b)
        end_time = time.time()
        duration = end_time - start_time

        # Assert result is returned
        assert "result" in result
        # Assert result shape is correct
        assert len(result["result"]) == size
        assert len(result["result"][0]) == size
        # Assert reasonable performance (under 10 seconds for 100x100)
        assert duration < 10.0

    def test_large_matrix_determinant(self):
        """Test matrix determinant with large matrix (50x50)."""
        size = 50
        # Create a large matrix (use identity for simplicity, det=1)
        matrix = np.eye(size).tolist()

        start_time = time.time()
        result = matrix_determinant(matrix)
        end_time = time.time()
        duration = end_time - start_time

        # Assert result is correct
        assert result == {"result": 1.0}
        # Assert reasonable performance
        assert duration < 5.0

    def test_large_vector_operations(self):
        """Test vector operations with large vectors (10000 elements)."""
        size = 10000
        vec_a = tuple(range(size))
        vec_b = tuple(range(size, 2*size))

        start_time = time.time()
        dot_result = vector_dot_product(vec_a, vec_b)
        mag_result = vector_magnitude(vec_a)
        end_time = time.time()
        duration = end_time - start_time

        # Assert results are correct
        expected_dot = sum(a*b for a, b in zip(vec_a, vec_b))
        assert dot_result == {"result": expected_dot}
        expected_mag = (sum(x**2 for x in vec_a)) ** 0.5
        assert mag_result["result"] == pytest.approx(expected_mag, rel=1e-10)
        # Assert reasonable performance
        assert duration < 2.0

    def test_large_summation(self):
        """Test summation with large range (1 to 100000)."""
        start_val = 1
        end_val = 100000

        start_time = time.time()
        result = summation("x", start_val, end_val)
        end_time = time.time()
        duration = end_time - start_time

        # Assert result is correct (sum of 1 to n = n(n+1)/2)
        expected = end_val * (end_val + 1) // 2
        assert result == {"result": expected}
        # Assert reasonable performance
        assert duration < 2.0