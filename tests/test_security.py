import pytest
from calculator_mcp_server import calculate


class TestEvalSandboxing:
    """Test cases for eval sandboxing to ensure safe execution."""

    def test_allowed_functions(self):
        """Test that allowed mathematical functions work."""
        result = calculate("sin(pi/2)")
        assert result == {"result": 1.0}

    def test_blocked_builtins(self):
        """Test that dangerous builtins are blocked."""
        result = calculate("__import__('os')")
        assert "error" in result
        assert "name '__import__' is not defined" in result["error"]

    def test_blocked_open(self):
        """Test that file operations are blocked."""
        result = calculate("open('/etc/passwd')")
        assert "error" in result
        assert "name 'open' is not defined" in result["error"]

    def test_blocked_exec(self):
        """Test that exec is blocked."""
        result = calculate("exec('print(1)')")
        assert "error" in result
        assert "name 'exec' is not defined" in result["error"]

    def test_blocked_eval(self):
        """Test that nested eval is blocked."""
        result = calculate("eval('1+1')")
        assert "error" in result
        assert "name 'eval' is not defined" in result["error"]

    def test_blocked_globals_access(self):
        """Test that accessing globals is blocked."""
        result = calculate("globals()")
        assert "error" in result
        assert "name 'globals' is not defined" in result["error"]


class TestInputValidation:
    """Test cases for input validation."""

    def test_valid_mathematical_expression(self):
        """Test that valid expressions are accepted."""
        result = calculate("2 + 3 * 4")
        assert result == {"result": 14}

    def test_invalid_characters(self):
        """Test that expressions with invalid characters are rejected."""
        result = calculate("2 + 3; print('hack')")
        assert "error" in result

    def test_code_injection_attempt(self):
        """Test that code injection attempts are blocked."""
        result = calculate("__builtins__.__import__('os').system('ls')")
        assert "error" in result

    def test_only_allowed_variables(self):
        """Test that only allowed variables are accessible."""
        result = calculate("x + 1")
        assert "error" in result
        assert "name 'x' is not defined" in result["error"]

    def test_empty_expression(self):
        """Test handling of empty expressions."""
        result = calculate("")
        assert "error" in result

    def test_whitespace_only(self):
        """Test handling of whitespace-only expressions."""
        result = calculate("   ")
        assert "error" in result


class TestSecurityVulnerabilities:
    """Test cases for potential security vulnerabilities."""

    def test_long_expression(self):
        """Test that very long expressions are handled safely."""
        long_expr = "2 + " * 1000 + "2"
        result = calculate(long_expr)
        # Should either succeed or fail gracefully, not crash
        assert isinstance(result, dict)
        if "result" in result:
            assert result["result"] == 2002  # 1000 additions of 2 + 2

    def test_deeply_nested_expression(self):
        """Test deeply nested expressions."""
        nested = "(" * 100 + "2" + ")" * 100
        result = calculate(nested)
        assert result == {"result": 2}

    def test_special_characters(self):
        """Test expressions with special characters."""
        result = calculate("2 + 3; print('hack')")
        assert "error" in result  # Semicolons should cause syntax error

    def test_unicode_characters(self):
        """Test expressions with unicode characters."""
        result = calculate("2 + π")  # pi symbol
        assert "error" in result  # Should use 'pi' not π

    def test_sql_injection_like(self):
        """Test SQL-like injection attempts."""
        result = calculate("SELECT * FROM users")
        assert "error" in result

    def test_buffer_overflow_attempt(self):
        """Test potential buffer overflow with large numbers."""
        large_num = "10**" + str(10**6)
        result = calculate(large_num)
        # Should handle large exponents gracefully
        assert isinstance(result, dict)