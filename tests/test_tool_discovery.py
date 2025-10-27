import pytest
from calculator_mcp_server import app


class TestToolDiscovery:
    """Test cases for tool discovery and listing functionality."""

    def test_tool_manager_exists(self):
        """Test that the app has a tool manager."""
        assert hasattr(app, '_tool_manager')
        assert app._tool_manager is not None

    def test_tools_dictionary_exists(self):
        """Test that the tool manager has a tools dictionary."""
        assert hasattr(app._tool_manager, '_tools')
        assert isinstance(app._tool_manager._tools, dict)

    def test_tools_are_registered(self):
        """Test that tools are registered in the tool manager."""
        tools = app._tool_manager._tools
        assert len(tools) > 0, "No tools are registered"

    def test_expected_tools_present(self):
        """Test that all expected tools are present."""
        tools = app._tool_manager._tools
        expected_tools = [
            "calculate", "solve_equation", "differentiate", "integrate",
            "mean", "variance", "standard_deviation", "median", "mode",
            "correlation_coefficient", "linear_regression", "confidence_interval",
            "matrix_addition", "matrix_multiplication", "matrix_transpose",
            "matrix_determinant", "vector_dot_product", "vector_cross_product",
            "vector_magnitude", "plot_function", "summation", "expand", "factorize"
        ]

        for tool_name in expected_tools:
            assert tool_name in tools, f"Tool '{tool_name}' is not registered"

    def test_tool_objects_have_function(self):
        """Test that each registered tool has a callable function."""
        tools = app._tool_manager._tools
        for tool_name, tool_obj in tools.items():
            assert hasattr(tool_obj, 'fn'), f"Tool '{tool_name}' does not have 'fn' attribute"
            assert callable(tool_obj.fn), f"Tool '{tool_name}' fn is not callable"

    def test_tool_count_matches_expected(self):
        """Test that the number of registered tools matches expected count."""
        tools = app._tool_manager._tools
        expected_count = 23  # Based on the expected_tools list
        assert len(tools) == expected_count, f"Expected {expected_count} tools, but found {len(tools)}"

    def test_no_duplicate_tools(self):
        """Test that there are no duplicate tool names."""
        tools = app._tool_manager._tools
        tool_names = list(tools.keys())
        unique_names = set(tool_names)
        assert len(tool_names) == len(unique_names), "Duplicate tool names found"

    def test_tool_names_are_strings(self):
        """Test that all tool names are strings."""
        tools = app._tool_manager._tools
        for tool_name in tools.keys():
            assert isinstance(tool_name, str), f"Tool name '{tool_name}' is not a string"
            assert len(tool_name.strip()) > 0, f"Tool name '{tool_name}' is empty or whitespace"

    def test_tools_are_not_empty_after_initialization(self):
        """Test that tools remain registered after app initialization."""
        # This test ensures tools aren't cleared during initialization
        tools = app._tool_manager._tools
        assert len(tools) > 0, "Tools were cleared during initialization"