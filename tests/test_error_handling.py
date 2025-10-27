import pytest
from unittest.mock import patch, MagicMock
from calculator_mcp_server import app
import numpy as np
import sympy as sp


class TestMCPLevelErrorResponses:
    """Test cases for MCP-level error responses."""

    def test_nonexistent_tool_call(self):
        """Test calling a non-existent tool returns appropriate error."""
        # Since FastMCP handles tool calls, we test by checking if tool exists
        assert "nonexistent_tool" not in app._tool_manager._tools

    def test_tool_call_with_invalid_parameters(self):
        """Test tool call with invalid parameter types."""
        # Test calculate with None
        result = app._tool_manager._tools["calculate"].fn(None)
        assert "error" in result

    def test_tool_call_with_missing_parameters(self):
        """Test tool call with missing required parameters."""
        # Most tools have required params, but let's test mean with no args
        # Actually, mean expects a list, so passing nothing would be invalid
        # Since it's Python, it will raise TypeError, but the tool should catch it
        with pytest.raises(TypeError):
            app._tool_manager._tools["mean"].fn()

    def test_invalid_jsonrpc_message_structure(self):
        """Test handling of invalid JSON-RPC message structure."""
        # Since FastMCP abstracts this, we test by simulating invalid inputs
        # For example, passing invalid types to tools
        result = app._tool_manager._tools["calculate"].fn(123)  # int instead of str
        assert "error" in result


class TestMalformedInputs:
    """Test cases for malformed inputs to tools."""

    def test_calculate_with_empty_string(self):
        """Test calculate with empty string."""
        result = app._tool_manager._tools["calculate"].fn("")
        assert "error" in result

    def test_calculate_with_none(self):
        """Test calculate with None input."""
        result = app._tool_manager._tools["calculate"].fn(None)
        assert "error" in result

    def test_solve_equation_with_no_equals(self):
        """Test solve_equation with malformed equation (no =)."""
        result = app._tool_manager._tools["solve_equation"].fn("x**2 + 1")
        assert "error" in result

    def test_solve_equation_with_multiple_equals(self):
        """Test solve_equation with multiple equals signs."""
        result = app._tool_manager._tools["solve_equation"].fn("x = 1 = 2")
        assert "error" in result

    def test_mean_with_string_instead_of_list(self):
        """Test mean with string instead of list."""
        result = app._tool_manager._tools["mean"].fn("not a list")
        assert "error" in result

    def test_mean_with_none_values_in_list(self):
        """Test mean with None values in list."""
        result = app._tool_manager._tools["mean"].fn([1, None, 3])
        assert "error" in result

    def test_matrix_addition_with_non_list(self):
        """Test matrix_addition with non-list inputs."""
        result = app._tool_manager._tools["matrix_addition"].fn("not a matrix", [[1, 2]])
        assert "error" in result

    def test_matrix_addition_with_inconsistent_rows(self):
        """Test matrix_addition with inconsistent row lengths."""
        result = app._tool_manager._tools["matrix_addition"].fn([[1, 2]], [[1]])
        assert "error" in result

    def test_vector_dot_product_with_strings(self):
        """Test vector_dot_product with string inputs."""
        result = app._tool_manager._tools["vector_dot_product"].fn("a", "b")
        assert "error" in result

    def test_vector_cross_product_with_2d_vectors(self):
        """Test vector_cross_product with 2D vectors (should fail)."""
        result = app._tool_manager._tools["vector_cross_product"].fn((1, 2), (3, 4))
        assert "error" in result

    def test_differentiate_with_empty_expression(self):
        """Test differentiate with empty expression."""
        result = app._tool_manager._tools["differentiate"].fn("")
        assert "error" in result

    def test_integrate_with_invalid_expression(self):
        """Test integrate with invalid expression."""
        result = app._tool_manager._tools["integrate"].fn("invalid")
        assert "error" in result

    def test_plot_function_with_non_string(self):
        """Test plot_function with non-string expression."""
        result = app._tool_manager._tools["plot_function"].fn(123)
        assert "error" in result

    def test_summation_with_non_integer_bounds(self):
        """Test summation with non-integer bounds."""
        result = app._tool_manager._tools["summation"].fn("x", "a", "b")
        assert "error" in result


class TestExceptionPropagation:
    """Test cases for exception propagation and error handling."""

    @patch('numpy.mean')
    def test_mean_numpy_exception_propagation(self, mock_mean):
        """Test that numpy exceptions in mean are caught and returned as errors."""
        mock_mean.side_effect = ValueError("Test error")
        result = app._tool_manager._tools["mean"].fn([1, 2, 3])
        assert "error" in result
        assert "Test error" in result["error"]

    @patch('scipy.stats.mode')
    def test_mode_scipy_exception_propagation(self, mock_mode):
        """Test that scipy exceptions in mode are caught."""
        mock_mode.side_effect = Exception("Scipy error")
        result = app._tool_manager._tools["mode"].fn([1, 2, 2])
        assert "error" in result

    @patch('calculator_mcp_server.solve')
    def test_solve_equation_sympy_exception_propagation(self, mock_solve):
        """Test that sympy exceptions in solve_equation are caught."""
        mock_solve.side_effect = sp.SympifyError("Invalid expression")
        result = app._tool_manager._tools["solve_equation"].fn("x = 1")
        assert "error" in result

    @patch('calculator_mcp_server.diff')
    def test_differentiate_sympy_exception_propagation(self, mock_diff):
        """Test that sympy exceptions in differentiate are caught."""
        mock_diff.side_effect = Exception("Sympy diff error")
        result = app._tool_manager._tools["differentiate"].fn("x**2")
        assert "error" in result

    @patch('calculator_mcp_server.sympy_integrate')
    def test_integrate_sympy_exception_propagation(self, mock_integrate):
        """Test that sympy exceptions in integrate are caught."""
        mock_integrate.side_effect = Exception("Sympy integrate error")
        result = app._tool_manager._tools["integrate"].fn("x")
        assert "error" in result

    @patch('numpy.linalg.det')
    def test_matrix_determinant_numpy_exception_propagation(self, mock_det):
        """Test that numpy exceptions in matrix_determinant are caught."""
        mock_det.side_effect = np.linalg.LinAlgError("Singular matrix")
        result = app._tool_manager._tools["matrix_determinant"].fn([[1, 2], [2, 4]])
        assert "error" in result

    @patch('matplotlib.pyplot.show')
    def test_plot_function_matplotlib_exception_propagation(self, mock_show):
        """Test that matplotlib exceptions in plot_function are caught."""
        mock_show.side_effect = Exception("Display error")
        result = app._tool_manager._tools["plot_function"].fn("x**2")
        assert "error" in result

    def test_calculate_eval_exception_propagation(self):
        """Test that eval exceptions in calculate are caught."""
        # This should trigger NameError
        result = app._tool_manager._tools["calculate"].fn("undefined_variable")
        assert "error" in result
        assert "name 'undefined_variable' is not defined" in result["error"]

    def test_calculate_syntax_error_propagation(self):
        """Test that syntax errors in calculate are caught."""
        result = app._tool_manager._tools["calculate"].fn("2 +")
        assert "error" in result

    def test_correlation_coefficient_length_mismatch(self):
        """Test correlation_coefficient with mismatched lengths."""
        result = app._tool_manager._tools["correlation_coefficient"].fn([1, 2], [1])
        assert "error" in result

    def test_linear_regression_insufficient_points(self):
        """Test linear_regression with insufficient points."""
        result = app._tool_manager._tools["linear_regression"].fn([(1, 1)])
        assert "error" in result

    def test_confidence_interval_invalid_confidence(self):
        """Test confidence_interval with invalid confidence level."""
        result = app._tool_manager._tools["confidence_interval"].fn([1, 2, 3], 1.5)
        assert "error" in result