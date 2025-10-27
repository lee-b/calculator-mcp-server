import pytest
import math
from calculator_mcp_server import (
    calculate,
    solve_equation,
    differentiate,
    integrate,
    expand,
    factorize,
    summation,
)


class TestCalculate:
    """Test cases for the calculate function."""

    def test_basic_arithmetic(self):
        result = calculate("2 + 3")
        assert result == {"result": 5}

    def test_multiplication(self):
        result = calculate("4 * 5")
        assert result == {"result": 20}

    def test_division(self):
        result = calculate("10 / 2")
        assert result == {"result": 5.0}

    def test_power(self):
        result = calculate("2 ** 3")
        assert result == {"result": 8}

    def test_modulo(self):
        result = calculate("7 % 3")
        assert result == {"result": 1}

    def test_trigonometric_sin(self):
        result = calculate("sin(pi/2)")
        assert result["result"] == pytest.approx(1.0, rel=1e-10)

    def test_trigonometric_cos(self):
        result = calculate("cos(0)")
        assert result["result"] == pytest.approx(1.0, rel=1e-10)

    def test_exponential(self):
        result = calculate("exp(1)")
        assert result["result"] == pytest.approx(math.e, rel=1e-10)

    def test_logarithm(self):
        result = calculate("log(100)")
        assert result["result"] == pytest.approx(4.605170185988092, rel=1e-10)

    def test_log10(self):
        result = calculate("log10(100)")
        assert result["result"] == 2.0

    def test_sqrt(self):
        result = calculate("sqrt(16)")
        assert result["result"] == 4.0

    def test_factorial(self):
        result = calculate("factorial(5)")
        assert result["result"] == 120

    def test_constants_pi(self):
        result = calculate("pi")
        assert result["result"] == pytest.approx(math.pi, rel=1e-10)

    def test_constants_e(self):
        result = calculate("e")
        assert result["result"] == pytest.approx(math.e, rel=1e-10)

    def test_complex_expression(self):
        result = calculate("2 * sin(pi/4) + sqrt(4)")
        assert result["result"] == pytest.approx(2 * math.sin(math.pi/4) + 2, rel=1e-10)

    def test_error_invalid_expression(self):
        result = calculate("invalid * expression")
        assert "error" in result
        assert "name 'invalid' is not defined" in result["error"]

    def test_error_syntax(self):
        result = calculate("2 +")
        assert "error" in result

    def test_error_division_by_zero(self):
        result = calculate("1 / 0")
        assert "error" in result

    def test_error_undefined_function(self):
        result = calculate("undefined_function(1)")
        assert "error" in result


class TestSolveEquation:
    """Test cases for the solve_equation function."""

    def test_linear_equation(self):
        result = solve_equation("2*x + 3 = 7")
        assert result == {"solutions": "[2]"}

    def test_quadratic_equation(self):
        result = solve_equation("x**2 - 5*x + 6 = 0")
        assert result == {"solutions": "[2, 3]"}

    def test_trigonometric_equation(self):
        result = solve_equation("sin(x) = 0.5")
        # Solutions should contain pi/6 and 5*pi/6
        assert "solutions" in result
        assert result["solutions"] != "[]"

    def test_no_solution(self):
        result = solve_equation("x = x + 1")
        assert result == {"solutions": "[]"}

    def test_multiple_variables_error(self):
        result = solve_equation("x + y = 5")
        assert "error" in result

    def test_invalid_equation_no_equals(self):
        result = solve_equation("x**2 + 1")
        assert "error" in result

    def test_complex_equation(self):
        result = solve_equation("x**3 - 6*x**2 + 11*x - 6 = 0")
        assert result == {"solutions": "[1, 2, 3]"}


class TestDifferentiate:
    """Test cases for the differentiate function."""

    def test_polynomial(self):
        result = differentiate("x**2")
        assert result == {"result": "2*x"}

    def test_polynomial_with_coefficients(self):
        result = differentiate("3*x**3 + 2*x**2 - x + 5")
        assert result == {"result": "9*x**2 + 4*x - 1"}

    def test_trigonometric(self):
        result = differentiate("sin(x)")
        assert result == {"result": "cos(x)"}

    def test_exponential(self):
        result = differentiate("exp(x)")
        assert result == {"result": "exp(x)"}

    def test_logarithmic(self):
        result = differentiate("log(x)")
        assert result == {"result": "1/x"}

    def test_product_rule(self):
        result = differentiate("x*sin(x)")
        assert result == {"result": "x*cos(x) + sin(x)"}

    def test_different_variable(self):
        result = differentiate("x*y", "y")
        assert result == {"result": "x"}

    def test_constant(self):
        result = differentiate("5")
        assert result == {"result": "0"}

    def test_error_invalid_expression(self):
        result = differentiate("invalid")
        assert "error" in result


class TestIntegrate:
    """Test cases for the integrate function."""

    def test_polynomial(self):
        result = integrate("x**2")
        assert result == {"result": "x**3/3"}

    def test_trigonometric(self):
        result = integrate("sin(x)")
        assert result == {"result": "-cos(x)"}

    def test_exponential(self):
        result = integrate("exp(x)")
        assert result == {"result": "exp(x)"}

    def test_logarithmic(self):
        result = integrate("1/x")
        assert result == {"result": "log(x)"}

    def test_different_variable(self):
        result = integrate("x*y", "y")
        assert result == {"result": "x*y**2/2"}

    def test_constant(self):
        result = integrate("5")
        assert result == {"result": "5*x"}

    def test_error_invalid_expression(self):
        result = integrate("invalid")
        assert "error" in result


class TestExpand:
    """Test cases for the expand function."""

    def test_binomial(self):
        result = expand("(x + 1)**2")
        assert result == {"result": "x**2 + 2*x + 1"}

    def test_trinomial(self):
        result = expand("(x + y)**3")
        assert result == {"result": "x**3 + 3*x**2*y + 3*x*y**2 + y**3"}

    def test_already_expanded(self):
        result = expand("x**2 + 2*x + 1")
        assert result == {"result": "x**2 + 2*x + 1"}

    def test_error_invalid_expression(self):
        result = expand("invalid")
        assert "error" in result


class TestFactorize:
    """Test cases for the factorize function."""

    def test_perfect_square(self):
        result = factorize("x**2 + 2*x + 1")
        assert result == {"result": "(x + 1)**2"}

    def test_quadratic(self):
        result = factorize("x**2 - 5*x + 6")
        assert result == {"result": "(x - 2)*(x - 3)"}

    def test_already_factored(self):
        result = factorize("(x + 1)**2")
        assert result == {"result": "(x + 1)**2"}

    def test_error_invalid_expression(self):
        result = factorize("invalid")
        assert "error" in result


class TestSummation:
    """Test cases for the summation function."""

    def test_simple_sum(self):
        result = summation("x", 1, 5)
        assert result == {"result": 15}

    def test_quadratic_sum(self):
        result = summation("x**2", 1, 3)
        assert result == {"result": 14}  # 1^2 + 2^2 + 3^2 = 1 + 4 + 9 = 14

    def test_constant_sum(self):
        result = summation("5", 1, 10)
        assert result == {"result": 50}

    def test_zero_start(self):
        result = summation("x", 0, 10)
        assert result == {"result": 55}

    def test_single_term(self):
        result = summation("x**2", 5, 5)
        assert result == {"result": 25}

    def test_error_invalid_expression(self):
        result = summation("invalid", 1, 5)
        assert "error" in result

    def test_large_range(self):
        result = summation("1", 1, 100)
        assert result == {"result": 100}