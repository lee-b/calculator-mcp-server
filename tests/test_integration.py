import pytest
import pytest_asyncio
import asyncio
import json
import logging
from mcp.client.stdio import stdio_client, StdioServerParameters
from calculator_mcp_server import app
from typing import Dict, Any

logger = logging.getLogger(__name__)


@pytest_asyncio.fixture
async def mcp_session():
    """Fixture providing an MCP client session connected to the server."""
    logger.info("Starting mcp_session fixture")
    server_params = StdioServerParameters(command="uv", args=["run", "calculator-mcp-server", "--stdio"])
    try:
        async with stdio_client(server_params) as (read_stream, write_stream):
            logger.info("stdio_client entered")
            from mcp.client.session import ClientSession
            try:
                async with ClientSession(read_stream, write_stream) as session:
                    logger.info("ClientSession entered")
                    await session.initialize()
                    logger.info("Session initialized, yielding")
                    yield session
                    logger.info("Session yielded back")
            except RuntimeError as e:
                if "Attempted to exit cancel scope in a different task" in str(e):
                    logger.warning(f"Ignoring known MCP teardown issue: {e}")
                else:
                    logger.error(f"ClientSession error: {e}")
                    raise
            except Exception as e:
                logger.error(f"ClientSession error: {e}")
                raise
            finally:
                logger.info("ClientSession exiting")
    except RuntimeError as e:
        if "Attempted to exit cancel scope in a different task" in str(e):
            logger.warning(f"Ignoring known MCP teardown issue: {e}")
        else:
            logger.error(f"stdio_client error: {e}")
            raise
    except Exception as e:
        logger.error(f"stdio_client error: {e}")
        raise
    finally:
        logger.info("stdio_client exiting")


class TestIntegration:
    """Integration tests for MCP server end-to-end functionality."""

    @pytest.mark.asyncio
    async def test_server_initialization(self, mcp_session):
        """Test that the server initializes correctly."""
        # Already initialized in fixture, just verify
        assert mcp_session is not None

    @pytest.mark.asyncio
    async def test_list_tools(self, mcp_session):
        """Test listing available tools."""
        tools_response = await mcp_session.list_tools()
        tools = tools_response.tools
        tool_names = [tool.name for tool in tools]

        # Check that expected tools are present
        expected_tools = [
            "calculate", "solve_equation", "differentiate", "integrate",
            "mean", "variance", "standard_deviation", "median", "mode",
            "correlation_coefficient", "linear_regression", "confidence_interval",
            "matrix_addition", "matrix_multiplication", "matrix_transpose",
            "matrix_determinant", "vector_dot_product", "vector_cross_product",
            "vector_magnitude", "plot_function", "summation", "expand", "factorize"
        ]

        for tool in expected_tools:
            assert tool in tool_names

    @pytest.mark.asyncio
    async def test_calculate_tool(self, mcp_session):
        """Test the calculate tool end-to-end."""
        result = await mcp_session.call_tool("calculate", {"expression": "2 + 2"})
        assert len(result.content) == 1
        assert result.content[0].type == "text"
        result_data = json.loads(result.content[0].text)
        assert result_data["result"] == 4

    @pytest.mark.asyncio
    async def test_calculate_tool_error(self, mcp_session):
        """Test the calculate tool with invalid expression."""
        result = await mcp_session.call_tool("calculate", {"expression": "invalid * expression"})
        assert len(result.content) == 1
        assert result.content[0].type == "text"
        result_data = json.loads(result.content[0].text)
        assert "error" in result_data

    @pytest.mark.asyncio
    async def test_solve_equation_tool(self, mcp_session):
        """Test the solve_equation tool end-to-end."""
        result = await mcp_session.call_tool("solve_equation", {"equation": "x**2 - 4 = 0"})
        assert len(result.content) == 1
        assert result.content[0].type == "text"
        result_data = json.loads(result.content[0].text)
        assert "solutions" in result_data

    @pytest.mark.asyncio
    async def test_mean_tool(self, mcp_session):
        """Test the mean tool end-to-end."""
        result = await mcp_session.call_tool("mean", {"data": [1.0, 2.0, 3.0, 4.0]})
        assert len(result.content) == 1
        assert result.content[0].type == "text"
        result_data = json.loads(result.content[0].text)
        assert result_data["result"] == 2.5

    @pytest.mark.asyncio
    async def test_matrix_addition_tool(self, mcp_session):
        """Test the matrix_addition tool end-to-end."""
        result = await mcp_session.call_tool("matrix_addition", {
            "matrix_a": [[1, 2], [3, 4]],
            "matrix_b": [[5, 6], [7, 8]]
        })
        assert len(result.content) == 1
        assert result.content[0].type == "text"
        result_data = json.loads(result.content[0].text)
        assert result_data["result"] == [[6, 8], [10, 12]]

    @pytest.mark.asyncio
    async def test_vector_dot_product_tool(self, mcp_session):
        """Test the vector_dot_product tool end-to-end."""
        result = await mcp_session.call_tool("vector_dot_product", {
            "vector_a": [1, 2],
            "vector_b": [3, 4]
        })
        assert len(result.content) == 1
        assert result.content[0].type == "text"
        result_data = json.loads(result.content[0].text)
        assert result_data["result"] == 11.0

    @pytest.mark.asyncio
    async def test_multiple_tool_calls(self, mcp_session):
        """Test multiple tool calls in sequence."""
        # First call
        result1 = await mcp_session.call_tool("calculate", {"expression": "3 * 4"})
        result_data1 = json.loads(result1.content[0].text)
        assert result_data1["result"] == 12

        # Second call
        result2 = await mcp_session.call_tool("mean", {"data": [10, 20, 30]})
        result_data2 = json.loads(result2.content[0].text)
        assert result_data2["result"] == 20.0

    @pytest.mark.asyncio
    async def test_tool_error_handling(self, mcp_session):
        """Test error handling for invalid tool calls."""
        # Test with empty data for mean
        result = await mcp_session.call_tool("mean", {"data": []})
        result_data = json.loads(result.content[0].text)
        assert "error" in result_data

        # Test with mismatched matrix dimensions
        result = await mcp_session.call_tool("matrix_addition", {
            "matrix_a": [[1, 2]],
            "matrix_b": [[3, 4, 5]]
        })
        result_data = json.loads(result.content[0].text)
        assert "error" in result_data