import pytest
import numpy as np
import warnings
from calculator_mcp_server import (
    mean,
    variance,
    standard_deviation,
    median,
    mode,
    correlation_coefficient,
    linear_regression,
    confidence_interval,
)


class TestMean:
    """Test cases for the mean function."""

    def test_basic_mean(self, sample_data):
        result = mean(sample_data)
        expected = np.mean(sample_data)
        assert result == {"result": float(expected)}

    def test_single_element(self, single_element_data):
        result = mean(single_element_data)
        assert result == {"result": 5.0}

    def test_empty_data_error(self, empty_data):
        result = mean(empty_data)
        assert "error" in result

    def test_large_data(self, large_data):
        result = mean(large_data)
        expected = np.mean(large_data)
        assert result == {"result": float(expected)}

    def test_negative_numbers(self):
        data = [-1, -2, -3, -4, -5]
        result = mean(data)
        assert result == {"result": -3.0}


class TestVariance:
    """Test cases for the variance function."""

    def test_basic_variance(self, sample_data):
        result = variance(sample_data)
        expected = np.var(sample_data)
        assert result == {"result": float(expected)}

    def test_single_element(self, single_element_data):
        result = variance(single_element_data)
        assert result == {"result": 0.0}

    def test_empty_data_error(self, empty_data):
        result = variance(empty_data)
        assert "error" in result


class TestStandardDeviation:
    """Test cases for the standard_deviation function."""

    def test_basic_std(self, sample_data):
        result = standard_deviation(sample_data)
        expected = np.std(sample_data)
        assert result == {"result": float(expected)}

    def test_single_element(self, single_element_data):
        result = standard_deviation(single_element_data)
        assert result == {"result": 0.0}

    def test_empty_data_error(self, empty_data):
        result = standard_deviation(empty_data)
        assert "error" in result


class TestMedian:
    """Test cases for the median function."""

    def test_odd_length(self, sample_data):
        result = median(sample_data)
        expected = np.median(sample_data)
        assert result == {"result": float(expected)}

    def test_even_length(self):
        data = [1, 2, 3, 4]
        result = median(data)
        assert result == {"result": 2.5}

    def test_single_element(self, single_element_data):
        result = median(single_element_data)
        assert result == {"result": 5.0}

    def test_empty_data_error(self, empty_data):
        result = median(empty_data)
        assert "error" in result

    def test_unsorted_data(self):
        data = [3, 1, 4, 1, 5]
        result = median(data)
        assert result == {"result": 3.0}


class TestMode:
    """Test cases for the mode function."""

    def test_single_mode(self):
        data = [1, 2, 2, 3]
        result = mode(data)
        assert result == {"result": 2.0}

    def test_multiple_modes(self):
        data = [1, 1, 2, 2, 3]
        result = mode(data)
        # scipy.stats.mode returns the smallest mode
        assert result == {"result": 1.0}

    def test_all_same(self, single_element_data):
        data = [5, 5, 5]
        result = mode(data)
        assert result == {"result": 5.0}

    def test_empty_data_error(self, empty_data):
        result = mode(empty_data)
        assert result == {"error": "Cannot compute mode of empty array"}


class TestCorrelationCoefficient:
    """Test cases for the correlation_coefficient function."""

    def test_perfect_positive(self):
        x = [1, 2, 3, 4]
        y = [2, 4, 6, 8]
        result = correlation_coefficient(x, y)
        assert result == {"result": 1.0}

    def test_perfect_negative(self):
        x = [1, 2, 3, 4]
        y = [4, 3, 2, 1]
        result = correlation_coefficient(x, y)
        assert result == {"result": -1.0}

    def test_no_correlation(self):
        x = [1, 2, 3, 4]
        y = [1, 1, 1, 1]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="invalid value encountered in divide")
            result = correlation_coefficient(x, y)
        assert result["result"] == pytest.approx(0.0, abs=1e-10)

    def test_different_lengths_error(self):
        x = [1, 2, 3]
        y = [1, 2]
        result = correlation_coefficient(x, y)
        assert "error" in result

    def test_empty_data_error(self, empty_data):
        result = correlation_coefficient(empty_data, empty_data)
        assert "error" in result


class TestLinearRegression:
    """Test cases for the linear_regression function."""

    def test_basic_regression(self, sample_points):
        result = linear_regression(sample_points)
        # Expected slope = 1.5, intercept = 0.333...
        assert "slope" in result
        assert "intercept" in result
        assert result["slope"] == pytest.approx(1.5, rel=1e-10)
        assert result["intercept"] == pytest.approx(0.3333333333333335, rel=1e-10)

    def test_perfect_fit(self):
        points = [(0, 0), (1, 1), (2, 2)]
        result = linear_regression(points)
        assert result["slope"] == 1.0
        assert result["intercept"] == 0.0

    def test_single_point_error(self):
        points = [(1, 2)]
        result = linear_regression(points)
        assert "error" in result

    def test_empty_data_error(self):
        result = linear_regression([])
        assert "error" in result


class TestConfidenceInterval:
    """Test cases for the confidence_interval function."""

    def test_basic_interval(self, sample_data):
        result = confidence_interval(sample_data, 0.95)
        assert "confidence_interval" in result
        lower, upper = result["confidence_interval"]
        assert isinstance(lower, float)
        assert isinstance(upper, float)
        assert lower < upper

    def test_different_confidence(self, sample_data):
        result_95 = confidence_interval(sample_data, 0.95)
        result_99 = confidence_interval(sample_data, 0.99)
        lower_95, upper_95 = result_95["confidence_interval"]
        lower_99, upper_99 = result_99["confidence_interval"]
        # 99% interval should be wider
        assert (upper_99 - lower_99) > (upper_95 - lower_95)

    def test_single_element(self, single_element_data):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="One or more sample arguments is too small")
            result = confidence_interval(single_element_data, 0.95)
        assert "confidence_interval" in result

    def test_empty_data_error(self, empty_data):
        result = confidence_interval(empty_data, 0.95)
        assert "error" in result

    def test_invalid_confidence(self, sample_data):
        result = confidence_interval(sample_data, 1.5)
        assert "error" in result