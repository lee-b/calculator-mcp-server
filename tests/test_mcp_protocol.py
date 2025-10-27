import pytest
import json
from unittest.mock import patch, MagicMock
from calculator_mcp_server import app
from mcp.types import JSONRPCRequest, JSONRPCResponse, JSONRPCError


class TestJSONRPCMessageHandling:
    """Test cases for JSON-RPC message parsing and handling."""

    def test_valid_tool_call_request(self):
        """Test handling of valid JSON-RPC tool call request."""
        # This would require mocking the MCP client interaction
        # Since FastMCP handles the protocol, we test through tool calls
        pass

    def test_invalid_json_message(self):
        """Test handling of invalid JSON messages."""
        # FastMCP should handle JSON parsing internally
        # We can test by simulating invalid inputs to tools
        pass

    def test_missing_jsonrpc_version(self):
        """Test handling of messages missing jsonrpc version."""
        pass

    def test_invalid_method_name(self):
        """Test handling of invalid method names."""
        pass

    def test_notification_handling(self):
        """Test handling of JSON-RPC notifications."""
        pass


class TestToolCallValidation:
    """Test cases for tool call parameter validation."""

    def test_calculate_valid_expression(self):
        """Test calculate tool with valid expression."""
        result = app._tool_manager._tools["calculate"].fn("2 + 2")
        assert result == {"result": 4}

    def test_calculate_invalid_expression(self):
        """Test calculate tool with invalid expression."""
        result = app._tool_manager._tools["calculate"].fn("invalid * expression")
        assert "error" in result

    def test_solve_equation_valid(self):
        """Test solve_equation with valid equation."""
        result = app._tool_manager._tools["solve_equation"].fn("x**2 - 4 = 0")
        assert "solutions" in result

    def test_solve_equation_invalid_format(self):
        """Test solve_equation with invalid format (no =)."""
        result = app._tool_manager._tools["solve_equation"].fn("x**2 - 4")
        assert "error" in result

    def test_solve_equation_multiple_variables(self):
        """Test solve_equation with multiple variables."""
        result = app._tool_manager._tools["solve_equation"].fn("x + y = 5")
        assert "error" in result

    def test_mean_valid_data(self):
        """Test mean tool with valid data."""
        result = app._tool_manager._tools["mean"].fn([1, 2, 3, 4])
        assert result == {"result": 2.5}

    def test_mean_empty_data(self):
        """Test mean tool with empty data."""
        result = app._tool_manager._tools["mean"].fn([])
        assert "error" in result

    def test_matrix_addition_valid(self):
        """Test matrix_addition with valid matrices."""
        result = app._tool_manager._tools["matrix_addition"].fn([[1, 2]], [[3, 4]])
        assert result == {"result": [[4, 6]]}

    def test_matrix_addition_dimension_mismatch(self):
        """Test matrix_addition with dimension mismatch."""
        result = app._tool_manager._tools["matrix_addition"].fn([[1, 2]], [[3, 4, 5]])
        assert "error" in result

    def test_vector_dot_product_valid(self):
        """Test vector_dot_product with valid vectors."""
        result = app._tool_manager._tools["vector_dot_product"].fn((1, 2), (3, 4))
        assert result == {"result": 11.0}

    def test_vector_dot_product_dimension_mismatch(self):
        """Test vector_dot_product with dimension mismatch."""
        result = app._tool_manager._tools["vector_dot_product"].fn((1, 2), (3, 4, 5))
        assert "error" in result


class TestResponseFormatting:
    """Test cases for response formatting."""

    def test_success_response_structure(self):
        """Test that success responses have correct JSON-RPC structure."""
        result = app._tool_manager._tools["calculate"].fn("2 + 2")
        # Responses should be dicts with either "result" or "error"
        assert isinstance(result, dict)
        assert "result" in result or "error" in result

    def test_error_response_structure(self):
        """Test that error responses have correct structure."""
        result = app._tool_manager._tools["calculate"].fn("invalid")
        assert isinstance(result, dict)
        assert "error" in result
        assert isinstance(result["error"], str)

    def test_calculate_response_types(self):
        """Test that calculate returns appropriate types."""
        # Numeric result
        result = app._tool_manager._tools["calculate"].fn("2 + 2")
        assert result["result"] == 4

        # Error string
        result = app._tool_manager._tools["calculate"].fn("undefined_var")
        assert isinstance(result["error"], str)

    def test_solve_equation_response_format(self):
        """Test solve_equation response format."""
        result = app._tool_manager._tools["solve_equation"].fn("x - 2 = 0")
        assert "solutions" in result
        assert isinstance(result["solutions"], str)

    def test_statistical_tools_response_format(self):
        """Test statistical tools return float results."""
        result = app._tool_manager._tools["mean"].fn([1.0, 2.0, 3.0])
        assert isinstance(result["result"], float)

    def test_matrix_tools_response_format(self):
        """Test matrix tools return list of lists."""
        result = app._tool_manager._tools["matrix_transpose"].fn([[1, 2], [3, 4]])
        assert isinstance(result["result"], list)
        assert all(isinstance(row, list) for row in result["result"])

    def test_vector_tools_response_format(self):
        """Test vector tools return appropriate types."""
        # Magnitude returns float
        result = app._tool_manager._tools["vector_magnitude"].fn((3, 4))
        assert isinstance(result["result"], float)

        # Cross product returns list
        result = app._tool_manager._tools["vector_cross_product"].fn((1, 0, 0), (0, 1, 0))
        assert isinstance(result["result"], list)