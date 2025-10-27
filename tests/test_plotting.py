import pytest
from unittest.mock import patch, MagicMock
from calculator_mcp_server import plot_function


class TestPlotFunction:
    """Test cases for the plot_function."""

    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.subplots')
    @patch('sympy.lambdify')
    def test_basic_plot(self, mock_lambdify, mock_subplots, mock_show):
        # Mock the lambdify function
        mock_func = MagicMock()
        mock_func.return_value = [0, 1, 4, 9, 16]  # x^2 values
        mock_lambdify.return_value = mock_func

        # Mock subplots
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        result = plot_function("x**2")
        assert result == {"result": "Plot generated successfully."}

        # Verify that lambdify was called
        mock_lambdify.assert_called_once()

        # Verify that subplots was called
        mock_subplots.assert_called_once()

        # Verify that show was called
        mock_show.assert_called_once()

    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.subplots')
    @patch('sympy.lambdify')
    def test_plot_with_custom_range(self, mock_lambdify, mock_subplots, mock_show):
        mock_func = MagicMock()
        mock_func.return_value = [-1, 0, 1, 4, 9]
        mock_lambdify.return_value = mock_func

        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        result = plot_function("x**2", start=-2, end=2, step=5)
        assert result == {"result": "Plot generated successfully."}

    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.subplots')
    @patch('sympy.lambdify')
    def test_plot_trigonometric(self, mock_lambdify, mock_subplots, mock_show):
        import numpy as np
        mock_func = MagicMock()
        x_vals = np.linspace(0, 2*np.pi, 100)
        mock_func.return_value = np.sin(x_vals)
        mock_lambdify.return_value = mock_func

        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        result = plot_function("sin(x)", start=0, end=2*3.14159, step=100)
        assert result == {"result": "Plot generated successfully."}

    def test_invalid_expression_error(self):
        result = plot_function("invalid_expression")
        assert "error" in result

    def test_complex_expression_error(self):
        result = plot_function("x**2 + invalid")
        assert "error" in result

    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.subplots')
    @patch('sympy.lambdify')
    def test_plot_zero_range(self, mock_lambdify, mock_subplots, mock_show):
        mock_func = MagicMock()
        mock_func.return_value = [4]  # x^2 at x=2
        mock_lambdify.return_value = mock_func

        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        result = plot_function("x**2", start=2, end=2, step=1)
        assert result == {"result": "Plot generated successfully."}