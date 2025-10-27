import pytest
from unittest.mock import patch
from calculator_mcp_server import app


class TestAppCreation:
    """Test cases for FastMCP app creation."""

    def test_app_name(self):
        """Test that the app is created with the correct name."""
        assert app.name == "Mathematical Calculator"


class TestToolRegistration:
    """Test cases for tool registration."""

    def test_tools_registered(self):
        """Test that all tools are registered."""
        # Expected tools based on the decorators in __init__.py
        expected_tools = [
            "calculate",
            "solve_equation",
            "differentiate",
            "integrate",
            "mean",
            "variance",
            "standard_deviation",
            "median",
            "mode",
            "correlation_coefficient",
            "linear_regression",
            "confidence_interval",
            "matrix_addition",
            "matrix_multiplication",
            "matrix_transpose",
            "matrix_determinant",
            "vector_dot_product",
            "vector_cross_product",
            "vector_magnitude",
            "plot_function",
            "summation",
            "expand",
            "factorize",
        ]
        registered_tools = list(app._tool_manager._tools.keys())
        assert len(registered_tools) == len(expected_tools)
        for tool in expected_tools:
            assert tool in registered_tools


class TestServerStartup:
    """Test cases for server startup."""

    @patch.object(app, 'run')
    def test_server_startup_stdio(self, mock_run):
        """Test server startup with stdio transport."""
        from calculator_mcp_server import main
        import sys
        # Mock sys.argv to simulate --stdio
        original_argv = sys.argv
        sys.argv = ['test', '--stdio']
        try:
            main()
            mock_run.assert_called_once_with(transport="stdio")
        finally:
            sys.argv = original_argv

    @patch.object(app, 'run')
    def test_server_startup_sse(self, mock_run):
        """Test server startup with SSE transport (default)."""
        from calculator_mcp_server import main
        import sys
        # Mock sys.argv to simulate no args
        original_argv = sys.argv
        sys.argv = ['test']
        try:
            main()
            mock_run.assert_called_once_with(transport="sse", host="0.0.0.0", port=9191)
        finally:
            sys.argv = original_argv

    @patch.object(app, 'run')
    def test_server_startup_sse_custom_host(self, mock_run):
        """Test server startup with SSE transport and custom host."""
        from calculator_mcp_server import main
        import sys
        # Mock sys.argv to simulate --host argument
        original_argv = sys.argv
        sys.argv = ['test', '--host', '127.0.0.1']
        try:
            main()
            mock_run.assert_called_once_with(transport="sse", host="127.0.0.1", port=9191)
        finally:
            sys.argv = original_argv

    @patch.object(app, 'run')
    def test_server_startup_sse_custom_port(self, mock_run):
        """Test server startup with SSE transport and custom port."""
        from calculator_mcp_server import main
        import sys
        # Mock sys.argv to simulate --port argument
        original_argv = sys.argv
        sys.argv = ['test', '--port', '8080']
        try:
            main()
            mock_run.assert_called_once_with(transport="sse", host="0.0.0.0", port=8080)
        finally:
            sys.argv = original_argv

    @patch.object(app, 'run')
    def test_server_startup_sse_custom_host_and_port(self, mock_run):
        """Test server startup with SSE transport and custom host and port."""
        from calculator_mcp_server import main
        import sys
        # Mock sys.argv to simulate --host and --port arguments
        original_argv = sys.argv
        sys.argv = ['test', '--host', '127.0.0.1', '--port', '8080']
        try:
            main()
            mock_run.assert_called_once_with(transport="sse", host="127.0.0.1", port=8080)
        finally:
            sys.argv = original_argv